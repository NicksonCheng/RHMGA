import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from models.HGT import HGT
from dgl.nn import GATConv, RelGraphConv, GraphConv, HeteroGraphConv, WeightBasis
from dgl.heterograph import DGLBlock, DGLGraph

ntypes = {
    "heco_acm": ["paper", "author", "subject"],
    "heco_aminer": ["paper", "author", "reference"],
    "heco_freebase": ["movie", "author", "director", "writer"],
    "PubMed": ["Disease", "Gene", "Chemical", "Species"],
}


class RelGraphConvHetero(nn.Module):

    def __init__(self, in_dim, out_dim, rel_names, num_bases=None, weight=True, self_loop=True, activation=None, dropout=0.0):
        """R-GCN层（用于异构图）

        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param weight: bool, optional 是否进行线性变换，默认为True
        :param self_loop: 是否包括自环消息，默认为True
        :param activation: callable, optional 激活函数，默认为None
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.rel_names = rel_names
        self.self_loop = self_loop
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = HeteroGraphConv({rel: GraphConv(in_dim, out_dim, norm="right", weight=False, bias=False) for rel in rel_names})

        self.use_weight = weight
        if not num_bases:
            num_bases = len(rel_names)
        self.use_basis = weight and 0 < num_bases < len(rel_names)
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_dim, out_dim), num_bases, len(rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(rel_names), in_dim, out_dim))
                nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain("relu"))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, nn.init.calculate_gain("relu"))

    def forward(self, g, inputs):
        """
        :param g: DGLGraph 异构图
        :param inputs: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight  # (R, d_in, d_out)
            kwargs = {rel: {"weight": weight[i]} for i, rel in enumerate(self.rel_names)}
        else:
            kwargs = {}
        hs = self.conv(g, inputs, mod_kwargs=kwargs)  # Dict[ntype, (N_i, d_out)]
        for ntype in hs:
            if self.self_loop:
                hs[ntype] += torch.matmul(inputs[ntype], self.loop_weight)
            if self.activation:
                hs[ntype] = self.activation(hs[ntype])
            hs[ntype] = self.dropout(hs[ntype])
        return hs


class Module(nn.Module):
    def __init__(self, relations, in_dim, out_dim, num_heads, module_name, dataset):
        super(Module, self).__init__()
        self.module_name = module_name
        self.dataset = dataset
        self.relations = relations
        self.gcn_layers = GraphConv(
            in_feats=in_dim,
            out_feats=out_dim,
            weight=True,
            activation=F.elu,
            allow_zero_in_degree=True,
        )
        self.gat_layers = GATConv(
            in_feats=in_dim,
            out_feats=out_dim,
            num_heads=num_heads,
            feat_drop=0.1,
            attn_drop=0.4,
            activation=F.elu,
            allow_zero_in_degree=True,
        )
        if dataset == "heco_aminer" or dataset == "heco_freebase":
            self.rgcn_layer = RelGraphConv(in_dim, out_dim, len(self.relations), regularizer="basis", num_bases=2)
        else:
            self.rgcn_layer = RelGraphConvHetero(in_dim=in_dim, out_dim=out_dim, rel_names=relations)
        self.hgt_layer = HGT(
            in_dims={ntype: in_dim for ntype in ntypes},
            hidden_dim=out_dim,
            num_heads=num_heads,
            ntypes=ntypes[self.dataset],
            etypes=relations,
            predict_ntype=ntypes[self.dataset][0],
            num_layers=1,
        )

    def mask_edges_func(self, g: DGLGraph, mask_rate: float = 0.3):

        # num_edges = target_edges_indices.shape[0]
        if g.is_homogeneous:
            num_edges = g.num_edges()
            permutation = torch.randperm(num_edges).to(g.device)
            num_mask_edges = int(mask_rate * num_edges)
            mask_edges_indices = permutation[:num_mask_edges]
            g.remove_edges(mask_edges_indices)
            return g, mask_edges_indices
        for etype in g.canonical_etypes:
            num_edges = g.num_edges(etype)
            permutation = torch.randperm(num_edges).to(g.device)
            num_mask_edges = int(mask_rate * num_edges)
            if num_mask_edges == 0:
                continue
            mask_edges_indices = permutation[:num_mask_edges]
            keep_edges = permutation[num_mask_edges:]

            # remove_indices = target_edges_indices[mask_edges]
            g.remove_edges(mask_edges_indices, etype=etype)
        return g

    def forward(self, block, dst_ntype, dst_feat, src_feats, curr_mask_rate):
        hetero_module = ["RGCN_hetero", "HGT"]
        dst_rels = [rel_tuple for rel_tuple in self.relations if dst_ntype == rel_tuple[2]]
        if self.module_name in hetero_module:
            ## change block to heterograph
            ## change block to heterograph
            homo_etype = f"{dst_ntype}-{dst_ntype}"
            dst_feat_ids = block.dstnodes(dst_ntype)
            feats = {**src_feats}

            ## species-species edge will be zero
            if self.dataset != "PubMed" or block.num_edges(homo_etype) == 0:

                feats[dst_ntype] = dst_feat
            edge_dict = {}
            for rel_type in dst_rels:
                src, dst = block.edges(etype=rel_type)
                edge_dict[rel_type] = (src, dst)
            g_hetero = dgl.heterograph(edge_dict)
            g_hetero = self.mask_edges_func(g_hetero, curr_mask_rate)
            for ntype in g_hetero.ntypes:
                ntype_indices = torch.unique(g_hetero.nodes(ntype)).tolist()
                feats[ntype] = feats[ntype][ntype_indices]
            if self.module_name == "RGCN_hetero":
                z = self.rgcn_layer(g_hetero, feats)

            elif self.module_name == "HGT":
                z = self.hgt_layer(g_hetero, feats)
            z = z[dst_ntype][dst_feat_ids]
            att_sc = torch.ones(len(dst_rels))
            return z, att_sc
        ## homogeneous graph block
        num_dst = block.num_dst_nodes(dst_ntype)
        combined_src, combined_etype, combined_dst = [], [], []
        combined_feats = [dst_feat]
        # Merge neighbors from different relations in the current block
        shift_index = num_dst
        edge_dict = {}
        for rel_type in dst_rels:
            rel_idx = self.relations.index(rel_type)
            src_ntype = rel_type[0]
            src, dst = block.edges(etype=rel_type)
            edge_dict[rel_type] = (src, dst)
            src_indices = torch.unique(src).tolist()
            map_src_indices = {id: idx + shift_index for idx, id in enumerate(src_indices)}
            map_src = torch.tensor([map_src_indices[id.item()] for id in src]).to(src.device)
            combined_etype.append(torch.full((map_src.shape[0],), rel_idx).to(src.device))
            combined_src.append(map_src)
            combined_feats.append(src_feats[src_ntype][src_indices])
            combined_dst.append(dst)
            shift_index += len(src_indices)

        # Concatenate edges from all relation types
        combined_etype = torch.cat(combined_etype)
        combined_src = torch.cat(combined_src)
        combined_feats = torch.cat(combined_feats)
        combined_dst = torch.cat(combined_dst)

        # Create a homogeneous graph for the current layer
        g_homogeneous = dgl.graph((combined_src, combined_dst))
        g_homogeneous.ndata["feat"] = combined_feats
        g_homogeneous, mask_edges_indices = self.mask_edges_func(g_homogeneous, curr_mask_rate)

        mask = torch.ones(combined_etype.size(0), dtype=torch.bool)
        mask[mask_edges_indices] = False
        combined_etype = combined_etype[mask]
        # Apply the corresponding GAT layer
        if self.module_name == "GAT":
            z = self.gat_layers(g_homogeneous, combined_feats).flatten(1)
        elif self.module_name == "GCN":
            z = self.gcn_layers(g_homogeneous, combined_feats)
        elif self.module_name == "RGCN_homo":
            z = self.rgcn_layer(g_homogeneous, combined_feats, combined_etype)
        dst_z = z[:num_dst]
        att_sc = torch.ones(len(dst_rels))
        # print(dst_z.shape)
        return dst_z, att_sc
