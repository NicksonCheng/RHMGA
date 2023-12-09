"""Heterogeneous Graph Attention Network (HAN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GATConv
# from torch_geometric.nn import GATConv


class SemanticAttention(nn.Module):

    def __init__(self, in_dim, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)  # weight sum
        )
        return

    def forward(self, z_m):
        w = self.seq(z_m).mean(dim=0)  # N * M * dim -> N * M * 1 -> M * 1
        a_w = torch.softmax(w, dim=0)  # M * 1
        a_w = a_w.expand(z_m.shape[0], a_w.shape[0], a_w.shape[1])  # N * M * 1
        z = (z_m * a_w).sum(dim=1)  # N * dim
        return z


class HANLayer(nn.Module):

    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gats = nn.ModuleList([
            # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
            GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads,
                    feat_drop=dropout, attn_drop=dropout, activation=F.elu)
            for _ in range(num_metapaths)
        ])
        self.semantic_attention = SemanticAttention(in_dim=out_dim*num_heads)

    def forward(self, subgraphs, h):
        z_m = [gat(s_g, h).flatten(start_dim=1) for gat, s_g in zip(
            self.gats, subgraphs)]  # N * K*d_out for each metapath

        z_m = torch.stack(z_m, dim=1)  # N * M * K*d_out (4057,3,128)
        z = self.semantic_attention(z_m)  # N * K*d_iyt (4057, 128)

        return z


class HAN(nn.Module):

    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(HAN, self).__init__()
            self.han_layer = HANLayer(num_metapaths=num_metapaths, in_dim=in_dim, out_dim=hidden_dim,
                                  num_heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(
            in_features=hidden_dim*num_heads, out_features=out_dim)

    def forward(self, subgraphs, h):
        z = self.han_layer(subgraphs, h)
        z = self.linear(z)
        return z
