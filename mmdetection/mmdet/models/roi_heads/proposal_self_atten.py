import torch.nn as nn
import torch
import numpy as np
from mmdet.models.utils import build_linear_layer

#
# class RelationReasoning(nn.Module):
#     def __init__(self):
#         super(RelationReasoning, self).__init__()
#         self.Wk = build_linear_layer(
#             {'type': 'Linear'},
#             in_features=300,
#             out_features=32)
#         self.Wq = build_linear_layer(
#             {'type': 'Linear'},
#             in_features=300,
#             out_features=32)
#         self.Wv = build_linear_layer(
#             {'type': 'Linear'},
#             in_features=300,
#             out_features=32)
#         self.Wl = build_linear_layer(
#             {'type': 'Linear'},
#             in_features=32,
#             out_features=300)
#     def forward(self, word):
#         '''
#         Q: [batch_size, n_heads, len_q, d_k]
#         K: [batch_size, n_heads, len_k, d_k]
#         V: [batch_size, n_heads, len_v(=len_k), d_v]
#         attn_mask: [batch_size, n_heads, seq_len, seq_len]
#         '''
#         K = self.Wk(word)
#         Q = self.Wq(word)
#         V = self.Wv(word)
#         d_k = K.shape[-1]
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
#         context_l = self.Wl(context)
#         context_res = context_l + word
#         return context_res, attn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H*W
        x = x.view(B, C, H*W)
        x = x.transpose(-1,1)
        # qkv = self.qkv(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        return x