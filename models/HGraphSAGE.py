import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from dgl.nn.pytorch.conv import GATConv

# from models.gat import GATConv
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
        att_mp = a_w.mean(0).squeeze()
        return z, att_mp


class Schema_Relation_Network(nn.Module):
    def __init__(self, relations, hidden_dim, out_dim, num_heads, weight_T, status="encoder"):
        super(Schema_Relation_Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.weight_T = weight_T
        self.status = status
        self.gats = nn.ModuleDict(
            {
                # torch_geometric GATConv(in_channels=in_dim,out_channels=hidden_dim,heads=num_heads,dropout=dropout)
                rel_tuple[1]: GATConv(
                    in_feats=self.hidden_dim,
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

        self.semantic_attention = SemanticAttention(in_dim=out_dim * self.num_heads)

    def forward(
        self,
        rels_subgraphs,
        dst_ntype,
        dst_feat,
        src_feat,
    ):

        ## Linear Transformation to same dimension
        neighbors_feat = {}

        if dst_feat.shape[1] != (self.hidden_dim * self.num_heads):
            dst_feat = self.weight_T[dst_ntype](dst_feat)
        for ntype, feat in src_feat.items():
            if feat.shape[1] != (self.hidden_dim * self.num_heads):
                neighbors_feat[ntype] = self.weight_T[ntype](feat)
            else:
                neighbors_feat[ntype] = feat
        ## aggregate the neighbor based on the relation
        z_r = {}
        for rels_tuple, rel_graph in rels_subgraphs.items():
            src_ntype = rels_tuple[0]
            rel = rels_tuple[1]
            # z_r[rel] = self.gats[rel](rel_graph, neighbors_feat[src_ntype], dst_feat)
            z_r[rel] = self.gats[rel](rel_graph, (neighbors_feat[src_ntype], dst_feat)).flatten(start_dim=1)

        ## semantic aggregation with all relation-based embedding

        z_r = torch.stack(list(z_r.values()), dim=1)

        z, att_mp = self.semantic_attention(z_r)

        return z, att_mp


class HGraphSAGE(nn.Module):
    def __init__(
        self,
        relations,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads,
        num_out_heads,
        num_layer,
        weight_T,
        status,
    ):
        super(HGraphSAGE, self).__init__()
        self.ntypes_out_dim = out_dim
        self.num_layer = num_layer
        self.weight_T = weight_T
        self.layers = nn.ModuleList()
        self.ntype_decoder = nn.ModuleDict()
        if self.num_layer == 1:
            if status == "encoder":
                self.layers.append(Schema_Relation_Network(relations, hidden_dim, out_dim, num_out_heads, weight_T, status))
            else:
                for ntype, dim in out_dim.items():

                    self.ntype_decoder[ntype] = Schema_Relation_Network(relations, in_dim, dim, num_out_heads, weight_T, status)
                self.layers.append(self.ntype_decoder)
        else:

            ##等等再來寫
            self.layers.append(Schema_Relation_Network(relations, hidden_dim, out_dim, num_heads, weight_T))
            for i in range(self.num_layer):
                self.layers.append(Schema_Relation_Network(relations, hidden_dim, out_dim, num_heads, weight_T))
            self.layers.append(Schema_Relation_Network(relations, hidden_dim, out_dim, num_out_heads, weight_T))

    ## filte non_zero in_degree dst node
    def mask_edges_func(self, rels_subg, mask_rate=0.3):
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
        mask_edges = permutation[:num_mask_edges]
        keep_edges = permutation[num_mask_edges:]

        # remove_indices = target_edges_indices[mask_edges]

        rels_subg.remove_edges(mask_edges)

        return rels_subg

    def neighbor_sampling(self, graph, curr_layer, relations, dst_ntype, dst_feat, features, status, mask_rate=0.3):
        src_feat = {}
        rels_subgraphs = {}
        for rels_tuple in relations:
            if dst_ntype == rels_tuple[2]:

                src_ntype = rels_tuple[0]
                src_feat[src_ntype] = features[src_ntype]
                ## sepearate relation graph
                clone_g = graph.clone()
                rels_subg = clone_g[rels_tuple]
                if status == "edge_recons" and curr_layer == 1:
                    rels_subg = self.mask_edges_func(rels_subg, mask_rate=0.3)

                rels_subgraphs[rels_tuple] = rels_subg
        if curr_layer >= self.num_layer:
            if status == "decoder":
                # for ntype, feat in src_feat.items():
                #     print(ntype, feat.shape)
                dst_feat, att_mp = self.layers[curr_layer - 1][dst_ntype](rels_subgraphs, dst_ntype, dst_feat, src_feat)
            else:
                dst_feat, att_mp = self.layers[curr_layer - 1](rels_subgraphs, dst_ntype, dst_feat, src_feat)
            print(f"{status}:{dst_feat.shape}")
            return dst_feat, att_mp

        else:
            ## layer n-1 dst feat will be layer n src feat
            # for ntype, feat in src_ntype_feats.items():
            #     src_ntype_feats[ntype], att_mp = self.neighbor_sampling(graph, curr_layer + 1, relations, ntype, feat, status)
            # dst_feat, att_mp = self.layers[curr_layer - 1](rels_subgraphs, dst_ntype, dst_feat, src_ntype_feats)

            return dst_feat, att_mp

    def forward(self, graph, start_layer, relations, dst_ntype, mask_dst_feat, status="encoder", curr_mask_rate=0.3):
        z, att_mp = self.neighbor_sampling(graph, start_layer, relations, dst_ntype, mask_dst_feat, status, curr_mask_rate)
        return z, att_mp
