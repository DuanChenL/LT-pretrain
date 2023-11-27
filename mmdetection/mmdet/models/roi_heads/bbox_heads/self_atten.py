import torch.nn as nn
import torch
import numpy as np
from mmdet.models.utils import build_linear_layer


class RelationReasoning(nn.Module):
    def __init__(self, num_classes=1230):
        super(RelationReasoning, self).__init__()
        self.k = 4
        self.num_classes = num_classes
        self.K = list()
        self.Q = list()
        self.V = list()
        for i in range(self.k):
            self.K.append(build_linear_layer(
                {'type': 'Linear'},
                in_features=1024,
                out_features=1024 // 4).cuda())
            self.Q.append(build_linear_layer(
                {'type': 'Linear'},
                in_features=1024,
                out_features=1024 // 4).cuda())
            self.V.append(build_linear_layer(
                {'type': 'Linear'},
                in_features=1024 // 4,
                out_features=1024).cuda())
        self.O = build_linear_layer(
            {'type': 'Linear'},
            in_features=1024,
            out_features=1024 // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(1024, self.num_classes + 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        O = self.O(x)
        o = torch.zeros(x.shape).cuda()
        for i in range(self.k):
            K = self.K[i](x)
            Q = self.Q[i](x)
            V = self.V[i](O)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[-1])
            a = nn.Softmax(dim=-1)(scores)
            o += torch.matmul(a, V)
        o += x
        o = self.dropout(self.relu(o))
        cls_scores = self.fc_cls(o)
        return cls_scores
