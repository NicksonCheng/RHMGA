"""Heterogeneous Graph Attention Network (HAN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import GATConv
from dgl.ops import edge_softmax

# from torch_geometric.nn import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.seq = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))  # weight sum
        return

    def forward(self, z_m):
        w = self.seq(z_m).mean(dim=0)  # N * M * dim -> N * M * 1 -> M * 1
        a_w = torch.softmax(w, dim=0)  # M * 1
        a_w = a_w.expand(z_m.shape[0], a_w.shape[0], a_w.shape[1])  # N * M * 1
        z = (z_m * a_w).sum(dim=1)  # N * dim
        return z


class HeCoGATConv(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.0, negative_slope=0.01, activation=None):
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(self, g, feat_src, feat_dst):
        with g.local_scope():
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)
            el = (feat_src * attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (N_src, 1)
            er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(dim=-1)  # (N_dst, 1)
            g.srcdata.update({"ft": feat_src, "el": el})
            g.dstdata["er"] = er
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(g.edata.pop("e"))
            g.edata["a"] = edge_softmax(g, e)  # (E, 1)

            # 消息传递
            g.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            ret = g.dstdata["ft"]
            if self.activation:
                ret = self.activation(ret)
            return ret


class HANLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gats = nn.ModuleList(
            [
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.elu)
                for _ in range(num_metapaths)
            ]
        )
        self.semantic_attention = SemanticAttention(in_dim=out_dim * num_heads)

    def forward(self, subgraphs, h):
        z_m = [gat(s_g, h).flatten(start_dim=1) for gat, s_g in zip(self.gats, subgraphs)]  # N * K*d_out for each metapath
        z_m = torch.stack(z_m, dim=1)  # N * M * K*d_out (4057,3,128)
        z = self.semantic_attention(z_m)  # N * K*d_iyt (4057, 128)

        return z


class Metapath_Relation_Network(nn.Module):
    def __init__(
        self, num_metapaths, num_relations, hidden_dim, out_dim, num_han_layer, num_srn_layer, num_heads, num_out_heads, dropout, weight_T, enc_dec
    ):
        super(Metapath_Relation_Network, self).__init__()

        self.weight_T = weight_T
        if enc_dec == "encoder":
            self.han_out_dim = out_dim // num_heads
        else:
            self.han_out_dim = out_dim
        self.srns = nn.ModuleList(
            [
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                # GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads,
                #         feat_drop=dropout, attn_drop=dropout, activation=F.elu)
                HeCoGATConv(hidden_dim=hidden_dim, attn_drop=0.3, activation=F.elu)
                for _ in range(num_relations)
            ]
        )
        self.semantic_attention = SemanticAttention(in_dim=hidden_dim)

        self.han_layers = nn.ModuleList()
        if num_han_layer == 1:
            self.han_layers.append(
                HANLayer(num_metapaths=num_metapaths, in_dim=hidden_dim, out_dim=self.han_out_dim, num_heads=num_out_heads, dropout=dropout)
            )
        else:
            self.han_layers.append(HANLayer(num_metapaths=num_metapaths, in_dim=hidden_dim, out_dim=hidden_dim, num_heads=num_heads, dropout=dropout))
            for layer in range(1, num_han_layer - 1):
                self.han_layers.append(
                    HANLayer(num_metapaths=num_metapaths, in_dim=hidden_dim * num_heads, out_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                )
            self.han_layers.append(
                HANLayer(
                    num_metapaths=num_metapaths, in_dim=hidden_dim * num_heads, out_dim=self.han_out_dim, num_heads=num_out_heads, dropout=dropout
                )
            )

    def forward(self, mp_subgraphs, sc_subgraphs, dst_feat, feats, enc_dec="encoder"):
        ### SRN Module
        # Linear Transformation to same dimension
        if enc_dec == "encoder":
            dst_feat = self.weight_T[0](dst_feat)

        z_r = []
        neighbors_feats = list(feats.values())
        neighbors_feats = [self.weight_T[idx](feat) for idx, feat in enumerate(neighbors_feats)]
        # print(feats.keys())
        for i in range(len(sc_subgraphs)):
            z_r.append(self.srns[i](sc_subgraphs[i], neighbors_feats[i + 1], dst_feat))

        z_r = torch.stack(z_r, dim=1)

        z = self.semantic_attention(z_r)
        ### HAN Module
        z_han = z.clone()
        for han_layer in self.han_layers:
            z_han = han_layer(mp_subgraphs, z_han)
        return z_han
