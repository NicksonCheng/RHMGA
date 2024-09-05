import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
from dgl.nn.pytorch.conv import GATConv

# from models.gat import GATConv
from dgl.ops import edge_softmax
from dgl.heterograph import DGLBlock


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

        att_sc = a_w.mean(0).squeeze()
        return z, att_sc


# class CrossAttention(nn.Module):
#     def __init__(self, ):
class Schema_Relation_Network(nn.Module):
    def __init__(
        self,
        relations: list,
        in_dim: int,
        out_dim: int | dict,
        weight_T: nn.ModuleDict,
        num_heads: int,
        status: str,
        aggregator: str,
    ):
        super(Schema_Relation_Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_T = weight_T
        self.num_heads = num_heads
        self.aggregator = aggregator
        ## project all type of node into same dimension
        self.gats = nn.ModuleDict(
            {
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                rel_tuple[1]: GATConv(
                    in_feats=self.in_dim,
                    out_feats=self.out_dim,
                    num_heads=self.num_heads,
                    feat_drop=0.1,
                    attn_drop=0.4,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
                # rel_tuple[1]: HeCoGATConv(hidden_dim=self.hidden_dim, attn_drop=0.3, activation=F.elu)
                for rel_tuple in relations
            }
        )

        self.semantic_attention = SemanticAttention(in_dim=out_dim * num_heads)

    def forward(
        self,
        rels_subgraphs: dict,
        dst_ntype: str,
        dst_feat: torch.Tensor,
        src_feat: dict,
        status: str,
    ):
        ## Linear Transformation to same dimension
        neighbors_feat = {}

        # if dst_feat.shape[1] != self.hidden_dim * self.num_heads:
        #     dst_feat = self.weight_T[dst_ntype](dst_feat)

        # for ntype, feat in src_feat.items():
        #     if feat.shape[1] != self.hidden_dim:
        #         neighbors_feat[ntype] = self.weight_T[ntype](feat)
        #     else:
        #         neighbors_feat[ntype] = feat

        ## aggregate the neighbor based on the relation
        z_r = {}
        for rels_tuple, rel_graph in rels_subgraphs.items():
            src_ntype = rels_tuple[0]
            rel = rels_tuple[1]
            # z_r[rel] = self.gats[rel](rel_graph, neighbors_feat[src_ntype], dst_feat)

            z_r[rel] = self.gats[rel](rel_graph, (src_feat[src_ntype], dst_feat)).flatten(1)

        ## semantic aggregation with all relation-based embedding
        ## prevent only one dst node or only one relation
        z_r_key = list(z_r.keys())
        z_r = torch.stack(list(z_r.values()), dim=1)
        z, att_sc = None, None
        if self.aggregator == "mean":
            att_sc = torch.ones(z_r.shape[1])
            z = z_r.mean(dim=1)

        elif self.aggregator == "attention":
            z, att_sc = self.semantic_attention(z_r)
        if att_sc.dim() == 0:
            rel_att_sc = {z_r_key[0]: att_sc}
        else:
            rel_att_sc = {key: att_sc[i] for i, key in enumerate(z_r_key)}
        return z, rel_att_sc


class HGraphSAGE(nn.Module):
    def __init__(
        self,
        relations: list,
        in_dim: int,
        hidden_dim: int,
        ntype_out_dim: int | dict,
        num_layer: int,
        num_heads: int,
        num_out_heads: int,
        dropout: float,
        weight_T: nn.ModuleDict,
        status: str,
        aggregator: str,
    ):
        super(HGraphSAGE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.ntype_out_dim = ntype_out_dim
        self.num_layer = num_layer
        self.weight_T = weight_T
        self.layers = nn.ModuleList()
        ## because this is HGraphSAGE, the head,dimension problem should be reverse compared to tradictional multilayer GAT
        if num_layer == 1:
            self.layers.append(Schema_Relation_Network(relations, self.in_dim, self.hidden_dim, weight_T, num_out_heads, status, aggregator))
        else:
            self.layers.append(
                Schema_Relation_Network(relations, self.hidden_dim * num_heads, self.hidden_dim, weight_T, num_out_heads, status, aggregator)
            )
            for i in range(1, num_layer - 1):
                self.layers.append(
                    Schema_Relation_Network(relations, self.hidden_dim * num_heads, self.hidden_dim, weight_T, num_heads, status, aggregator)
                )
            self.layers.append(Schema_Relation_Network(relations, self.in_dim, self.hidden_dim, weight_T, num_heads, status, aggregator))

        ## this is for src_feat encoder
        if status == "encoder":
            extra_in_dim = self.in_dim if num_layer == 1 else self.hidden_dim * num_heads
            self.layers.append(Schema_Relation_Network(relations, extra_in_dim, self.hidden_dim, weight_T, num_heads, status, aggregator))

        ## out_dim has different type of nodes
        if status == "decoder":
            self.ntypes_decoder_trans = nn.ModuleDict({ntype: nn.Linear(hidden_dim, out_dim) for ntype, out_dim in self.ntype_out_dim.items()})

    ## filter non_zero in_degree dst node
    def mask_edges_func(self, rels_subg: DGLBlock, mask_rate: float = 0.3):
        # src_nodes, dst_nodes = rels_subg.edges()
        # in_degrees = rels_subg.in_degrees()

        # non_zero_in_degrees = torch.nonzero(in_degrees > 1).squeeze()
        # target_edges_indices = []

        # for nz_dst_node in non_zero_in_degrees:
        #     target_edges_indices.extend(torch.nonzero(dst_nodes == nz_dst_node).squeeze().tolist())
        # ## edge indices of non_zero in_degree dst node
        # target_edges_indices = torch.tensor(target_edges_indices).to(rels_subg.device)

        # num_edges = target_edges_indices.shape[0]
        num_edges = rels_subg.num_edges()
        permutation = torch.randperm(num_edges).to(rels_subg.device)
        num_mask_edges = int(mask_rate * num_edges)
        if num_mask_edges == 0:
            return rels_subg, None, None
        mask_edges = permutation[:num_mask_edges]
        keep_edges = permutation[num_mask_edges:]

        # remove_indices = target_edges_indices[mask_edges]
        rels_subg.remove_edges(mask_edges)
        return rels_subg, mask_edges, keep_edges

    def seperate_relation_graph(
        self, subg: DGLBlock, relations: list, dst_ntype: str, src_ntype_feats: torch.Tensor, masked: bool, mask_rate: float = 0.3
    ):
        rels_subgraphs = {}
        rels_masked_edge = {}
        rels_keep_edge = {}
        use_src_ntype_feats = {}
        for rels_tuple in relations:
            src, rel, dst = rels_tuple
            if dst_ntype == dst and subg.num_edges(rels_tuple) > 0:
                ## sepearate relation graph
                rels_subg = subg[rels_tuple]
                mask_edges, keep_edges = None, None
                use_src_ntype_feats[src] = src_ntype_feats[src]
                if masked:
                    rels_subg, mask_edges, keep_edges = self.mask_edges_func(rels_subg, mask_rate)
                rels_subgraphs[rels_tuple] = rels_subg
                if mask_edges != None:
                    rels_masked_edge[rel] = mask_edges
                    rels_keep_edge[rel] = keep_edges
        return rels_subgraphs, use_src_ntype_feats, rels_masked_edge, rels_keep_edge

    def neighbor_sampling(
        self,
        subgs: list,
        curr_layer: int,
        relations: list,
        dst_ntype: str,
        dst_feat: torch.Tensor,
        subgs_src_ntype_feats: list,
        status: str,
        curr_mask_rate: float = 0.3,
    ):

        att_sc = None
        edge_masked = True if status == "edge_recons_encoder" and curr_layer == 1 else False

        if status == "evaluation":
            curr_subg = subgs[0]
            src_ntype_feats = subgs_src_ntype_feats[0]
        else:
            curr_subg = subgs[-curr_layer]
            src_ntype_feats = subgs_src_ntype_feats[-curr_layer]

        rels_subgraphs, use_src_ntype_feats, rel_mask_edges, rel_keep_edges = self.seperate_relation_graph(
            curr_subg, relations, dst_ntype, src_ntype_feats, edge_masked, curr_mask_rate
        )
        if curr_layer < self.num_layer:
            ## layer n-1 dst feat will be layer n src feat
            for ntype, feat in use_src_ntype_feats.items():
                use_src_ntype_feats[ntype], att_sc, _, _ = self.neighbor_sampling(
                    subgs, curr_layer + 1, relations, ntype, feat, subgs_src_ntype_feats, status
                )
        dst_feat, att_sc = self.layers[curr_layer - 1](rels_subgraphs, dst_ntype, dst_feat, use_src_ntype_feats, status)

        if curr_layer == self.num_layer and status == "encoder":
            ## need to update the last layer src_feat
            next_layer = curr_layer + 1
            next_subg = subgs[-next_layer]

            next_subg_src_feats = subgs_src_ntype_feats[-next_layer]
            next_subg_dst_feats = subgs_src_ntype_feats[-curr_layer]
            for ntype, feat in next_subg_dst_feats.items():
                ## small batch size problem
                if feat.shape[0] == 0:
                    continue
                next_rels_subgraphs, use_src_ntype_feats, _, _ = self.seperate_relation_graph(
                    next_subg, relations, ntype, next_subg_src_feats, False, curr_mask_rate
                )
                ## current layer's src feat will be next layer's dst feat
                subgs_src_ntype_feats[-curr_layer][ntype], att_sc = self.layers[next_layer - 1](
                    next_rels_subgraphs, ntype, feat, use_src_ntype_feats, status
                )
        return dst_feat, att_sc, rel_mask_edges, rel_keep_edges

    def forward(
        self,
        subgs: list,
        start_layer: int,
        relations: list,
        dst_ntype: str,
        dst_feat: torch.Tensor,
        subgs_src_ntype_feats: list,
        status: str = "encoder",
        curr_mask_rate: float = 0.3,
    ):
        z, att_sc, rel_mask_edge, rel_keep_edge = self.neighbor_sampling(
            subgs, start_layer, relations, dst_ntype, dst_feat, subgs_src_ntype_feats, status, curr_mask_rate
        )
        if status == "decoder":
            z = self.ntypes_decoder_trans[dst_ntype](z)
        if status == "edge_recons_encoder":
            return z, att_sc, rel_mask_edge, rel_keep_edge
        else:
            return z, att_sc
