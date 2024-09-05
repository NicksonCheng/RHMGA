import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv
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
            # print(el.shape)
            # print(er.shape)
            # print("--------------------------")
            # print(f"src feat: {g.srcdata['feat'].shape}")
            # print(f"src feat: {g.dstdata['feat'].shape}")
            # print("--------------------------")
            g.srcdata.update({"ft": feat_src, "el": el})
            g.dstdata["er"] = er
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
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
        relations,
        hidden_dim,
        ntype_out_dim,
        num_heads,
        num_out_heads,
        num_layer,
        dropout,
        weight_T,
        status,
    ):
        super(Schema_Relation_Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.weight_T = weight_T

        self.gats = nn.ModuleDict(
            {
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                # rel_tuple[1]: GATConv(
                #     in_feats=hidden_dim,
                #     out_feats=hidden_dim,
                #     num_heads=1,
                #     feat_drop=dropout,
                #     attn_drop=dropout,
                #     activation=F.elu,
                # )
                rel_tuple[1]: HeCoGATConv(hidden_dim=self.hidden_dim, attn_drop=0.3, activation=F.elu)
                for rel_tuple in relations
            }
        )

        self.semantic_attention = SemanticAttention(in_dim=hidden_dim)

        ## out_dim has different type of nodes
        if status == "decoder":
            self.ntypes_decoder_trans = nn.ModuleDict({ntype: nn.Linear(hidden_dim, out_dim) for ntype, out_dim in ntype_out_dim.items()})

    def forward(
        self,
        rels_subgraphs,
        dst_ntype,
        dst_feat,
        src_feat,
        status="encoder",
    ):
        ## Linear Transformation to same dimension
        if status == "encoder":
            dst_feat = self.weight_T[dst_ntype](dst_feat)
        neighbors_feat = {ntype: self.weight_T[ntype](feat) for ntype, feat in src_feat.items()}
        ## aggregate the neighbor based on the relation
        z_r = {}
        for rels_tuple, rel_graph in rels_subgraphs.items():
            src_ntype = rels_tuple[0]
            rel = rels_tuple[1]
            z_r[rel] = self.gats[rel](rel_graph, neighbors_feat[src_ntype], dst_feat)
            # z_r[rel] = self.gats[rel](rel_graph, (neighbors_feat[src_ntype], dst_feat)).squeeze()
        ## semantic aggregation with all relation-based embedding
        ## if this is edge reconstruction, we do not used semantic aggreagation, just return the z_r
        z_r = torch.stack(list(z_r.values()), dim=1)
        z = self.semantic_attention(z_r)

        if status == "decoder":
            z = self.ntypes_decoder_trans[dst_ntype](z)
        return z
