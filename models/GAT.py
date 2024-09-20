import torch
import dgl
import torch.nn as nn
from dgl.nn import GATConv


class GATAggregator(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads):
        super(GATAggregator, self).__init__()
        # GAT Layer
        self.gat = GATConv(in_feats, hidden_feats, num_heads)

    def forward(self, blocks, feats):
        # Combine edges from different relation blocks
        combined_src, combined_dst, combined_edges = [], [], []

        # Loop over each relation type and add edges to a combined list
        for rel_key, block in blocks.items():
            src, dst = block.edges()
            combined_src.append(src)
            combined_dst.append(dst)

        # Concatenate all sources and destinations from different relations
        combined_src = torch.cat(combined_src)
        combined_dst = torch.cat(combined_dst)

        # Create a homogeneous graph with all edges
        g_homogeneous = dgl.graph((combined_src, combined_dst), num_nodes=blocks[("Nei", "Nei-paper", "paper")].num_dst_nodes)
        print(g_homogeneous)
        # Apply GAT on the homogeneous graph
        output_feats = self.gat(g_homogeneous, feats)

        return output_feats
