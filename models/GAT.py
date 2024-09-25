import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, RelGraphConv
class AggregateModule(nn.Module):
    def __init__(self, relations, in_dim, out_dim, num_heads,module_name):
        super(AggregateModule, self).__init__()
        self.module_name=module_name
        self.relations = relations
        self.gat_layers = GATConv(
            in_feats=in_dim,
            out_feats=out_dim,
            num_heads=num_heads,
            feat_drop=0.1,
            attn_drop=0.4,
            activation=F.elu,
            allow_zero_in_degree=True,
        )
        self.rgcn_layer = RelGraphConv(in_dim,out_dim,len(self.relations), regularizer='basis', num_bases=2)

    def forward(self, block, dst_ntype, dst_feat,src_feats,status):
        num_dst=block.num_dst_nodes(dst_ntype)
        combined_src, combined_etype, combined_dst = [], [], []
        combined_feats=[dst_feat]
        # Merge neighbors from different relations in the current block
        dst_rels = [rel_tuple for rel_tuple in self.relations if dst_ntype == rel_tuple[2]]

        shift_index=num_dst
        for rel_type in dst_rels:
            rel_idx=self.relations.index(rel_type)
            src_ntype=rel_type[0]
            src, dst = block.edges(etype=rel_type)
            src_indices=torch.unique(src).tolist()
            map_src_indices={id:idx+shift_index for idx,id in enumerate(src_indices)}
            map_src=torch.tensor([map_src_indices[id.item()] for id in src]).to(src.device)
            combined_etype.append(torch.full((map_src.shape[0],),rel_idx).to(src.device))
            combined_src.append(map_src)
            combined_feats.append(src_feats[src_ntype][src_indices])
            combined_dst.append(dst)
            shift_index+=len(src_indices)

        # Concatenate edges from all relation types
        #print(combined_src)

        combined_etype=torch.cat(combined_etype)
        combined_src = torch.cat(combined_src)
        combined_feats=torch.cat(combined_feats)
        combined_dst = torch.cat(combined_dst)
        
        # Create a homogeneous graph for the current layer
        g_homogeneous = dgl.graph((combined_src, combined_dst))
        g_homogeneous.ndata["feat"]=combined_feats
        # Apply the corresponding GAT layer
        if self.module_name=="GAT":
            z = self.gat_layers(g_homogeneous, combined_feats).flatten(1)
        elif self.module_name=="RGCN":
            z=self.rgcn_layer(g_homogeneous,combined_feats,combined_etype)
        dst_z=z[:num_dst]
        att_sc = torch.ones(len(dst_rels))
        #print(dst_z.shape)
        return dst_z,att_sc
