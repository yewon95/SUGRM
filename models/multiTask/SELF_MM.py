# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from models.subNets.transformers_encoder.sp_transformer import SPEncoder
from SEmodule import SELayer

__all__ = ['SELF_MM']

class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        self.audio_in, self.video_in = args.feature_dims[1:]

        self.out_dropout = args.out_dropout

        """SE module"""
        self.se_model = SELayer(self.text_model.model.pooler.dense.out_features, self.audio_in, self.video_in)

        """SPT"""
        self.proj1_f = nn.Linear(args.combined_dim*3, args.combined_dim)
        self.proj2_f = nn.Linear(args.combined_dim, args.combined_dim)
        self.out_layer_f = nn.Linear(args.combined_dim, 1)

        self.proj1_t = nn.Linear(args.combined_dim, args.combined_dim)
        self.proj2_t = nn.Linear(args.combined_dim, args.combined_dim)
        self.out_layer_t = nn.Linear(args.combined_dim, 1)

        self.proj1_a = nn.Linear(args.combined_dim, args.combined_dim)
        self.proj2_a = nn.Linear(args.combined_dim, args.combined_dim)
        self.out_layer_a = nn.Linear(args.combined_dim, 1)

        self.proj1_v = nn.Linear(args.combined_dim, args.combined_dim)
        self.proj2_v = nn.Linear(args.combined_dim, args.combined_dim)
        self.out_layer_v = nn.Linear(args.combined_dim, 1)

        # SPT
        self.input_dims = dict(t=args.orig_d_l,
                               a=args.orig_d_a,
                               v=args.orig_d_v)
        self.spe = SPEncoder(embed_dim=args.d_model,
                             input_dims=self.input_dims,
                             num_heads=args.num_heads,
                             layers=args.layers,
                             attn_dropout=args.attn_dropout,
                             relu_dropout=args.relu_dropout,
                             res_dropout=args.res_dropout,
                             embed_dropout=args.embed_dropout,
                             S=args.S, r=args.r,
                             shift_mode=args.shift_mode,
                             use_fast=args.use_fast,
                             use_dense=args.use_dense,
                             device='cuda')

    def transformer_model(self, type, args):
        if type == "T":
            embed_dim, attn_dropout, num_heads = self.text_model.model.pooler.dense.out_features, args.attn_dropout_t, args.num_heads_t
        elif type == "V":
            embed_dim, attn_dropout, num_heads = self.video_in, args.attn_dropout_v, args.num_heads_v
        elif type == "A":
            embed_dim, attn_dropout, num_heads = self.audio_in, args.attn_dropout_a, args.num_heads_a
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=args.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=args.relu_dropout,
                                  res_dropout=args.res_dropout,
                                  embed_dropout=args.embed_dropout,
                                  attn_mask=args.attn_mask)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video
        text = self.text_model(text)

        # SE module
        text, video, audio = self.se_model(text, audio, video)

        # sparse phased transformer
        h_a, h_t, h_v=self.spe(audio,text,video)
        audio, text, video = h_a[-1], h_t[-1], h_v[-1]
        fusion_h = torch.cat([text, audio, video], dim=-1)

        x_f = self.proj2_f(F.dropout(F.relu(self.proj1_f(fusion_h)), p=self.out_dropout, training=self.training))
        output_fusion = self.out_layer_f(x_f)

        x_t = self.proj2_t(F.dropout(F.relu(self.proj1_t(text)), p=self.out_dropout, training=self.training))
        output_text = self.out_layer_t(x_t)

        x_a = self.proj2_a(F.dropout(F.relu(self.proj1_a(audio)), p=self.out_dropout, training=self.training))
        output_audio = self.out_layer_a(x_a)

        x_v = self.proj2_v(F.dropout(F.relu(self.proj1_v(video)), p=self.out_dropout, training=self.training))
        output_video = self.out_layer_v(x_v)


        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': x_t,
            'Feature_a': x_a,
            'Feature_v': x_v,
            'Feature_f': x_f,
        }
        return res

