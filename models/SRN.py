import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GATConv
from dgl.ops import edge_softmax


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


## only cared about in_dim
class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )  # weight sum
        return

    def forward(self, z_m):
        w = self.seq(z_m).mean(dim=0)
        a_w = torch.softmax(w, dim=0)
        a_w = a_w.expand(z_m.shape[0], a_w.shape[0], a_w.shape[1])
        z = (z_m * a_w).sum(dim=1)
        return z


class Schema_Relation_Network(nn.Module):
    def __init__(
        self,
        num_relations,
        hidden_dim,
        out_dim,
        num_heads,
        num_out_heads,
        num_layer,
        dropout,
        weight_T,
        enc_dec,
    ):
        super(Schema_Relation_Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.weight_T = weight_T
        self.gats = nn.ModuleList(
            [
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                # GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads,
                #         feat_drop=dropout, attn_drop=dropout, activation=F.elu)
                HeCoGATConv(hidden_dim=self.hidden_dim, attn_drop=0.3, activation=F.elu)
                for _ in range(num_relations)
            ]
        )

        self.semantic_attention = SemanticAttention(in_dim=hidden_dim)
        self.decoder_trans = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph, relations, dst_ntype, dst_feat, features, enc_dec="encoder"):

        ## Linear Transformation to same dimension
        sc_subgraphs = {}

        for rels in relations:
            ## dst_ntype include in relaions, then take its neighbor ntype as subgraph
            if dst_ntype == rels[2]:
                ntype = rels[0]
                sc_subgraphs[ntype] = graph[rels]

        if enc_dec == "encoder":
            dst_feat = self.weight_T[dst_ntype](dst_feat)
        neighbors_feat = {
            ntype: self.weight_T[ntype](feat)
            for ntype, feat in features.items()
            if ntype != dst_ntype
        }
        ## aggregate the neighbor based on the relation
        z_r = []

        for i, (ntype, subgraph) in enumerate(sc_subgraphs.items()):
            z_r.append(self.gats[i](subgraph, neighbors_feat[ntype], dst_feat))
        ## semantic aggregation with all relation-based embedding
        z_r = torch.stack(z_r, dim=1)
        z = self.semantic_attention(z_r)
        if enc_dec == "decoder":
            z = self.decoder_trans(z)
        return z
