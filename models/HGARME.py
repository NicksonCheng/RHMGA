import torch
import torch.nn as nn
from models.HAN import HAN
from models.SRN import Schema_Relation_Network
from utils.evaluate import cosine_similarity, mse
from functools import partial
from dgl.transforms import DropEdge
from collections import Counter
import traceback
import datetime
import dgl


class MultiLayerPerception(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerPerception, self).__init__()
        self.hidden = input_dim * 2
        self.fc1 = nn.Linear(input_dim, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def module_selection(
    relations,
    in_dim,
    hidden_dim,
    out_dim,
    num_heads,
    num_out_heads,
    num_layer,
    dropout,
    module_type,
    weight_T,
    enc_dec,
):
    if module_type == "HAN":
        return HAN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layer=num_layer,
            num_heads=num_heads,
            num_out_heads=num_out_heads,
            dropout=dropout,
        )
    elif module_type == "SRN":
        return Schema_Relation_Network(
            relations=relations,
            hidden_dim=hidden_dim,
            ntype_out_dim=out_dim,
            num_layer=num_layer,
            num_heads=num_heads,
            num_out_heads=num_out_heads,
            dropout=dropout,
            weight_T=weight_T,
            enc_dec=enc_dec,
        )
    elif module_type == "MLP":
        return MultiLayerPerception(input_dim=in_dim, output_dim=out_dim)


class HGARME(nn.Module):
    def __init__(self, relations, target_type, all_types, target_in_dim, ntype_in_dim, args):
        super(HGARME, self).__init__()
        self.mask_rate = args.mask_rate
        self.target_in_dim = target_in_dim
        self.ntype_in_dim = ntype_in_dim
        self.target_type = target_type
        self.all_types = all_types

        self.hidden_dim = args.num_hidden
        self.num_layer = args.num_layer
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.dropout = args.dropout
        self.gamma = args.gamma
        self.edge_recons = args.edge_recons
        self.feat_recons = args.feat_recons
        self.all_feat_recons = args.all_feat_recons
        self.all_edge_recons = args.all_edge_recons
        # encoder/decoder hidden dimension
        if self.encoder_type == "HAN":
            self.enc_dim = self.hidden_dim // self.num_heads
        else:
            self.enc_dim = self.hidden_dim
        self.enc_heads = self.num_heads

        self.dec_in_dim = self.hidden_dim  # enc_dim * enc_heads
        if self.decoder_type == "HAN":
            self.dec_hidden_dim = self.hidden_dim // self.num_heads
        else:
            self.dec_hidden_dim = self.hidden_dim
        self.dec_heads = self.num_out_heads

        ## project all type of node into same dimension
        self.weight_T = nn.ModuleDict({t: nn.Linear(self.ntype_in_dim[t], self.hidden_dim) for t in self.all_types})
        self.encoder = module_selection(
            relations=relations,
            in_dim=self.target_in_dim,
            hidden_dim=self.enc_dim,
            out_dim=self.enc_dim,
            num_heads=self.enc_heads,
            num_out_heads=self.enc_heads,
            num_layer=self.num_layer,
            dropout=self.dropout,
            module_type=self.encoder_type,
            weight_T=self.weight_T,
            enc_dec="encoder",
        )
        # linear transformation from encoder to decoder
        self.initial_enc_dec_dim(relations=relations)
        self.decoder = module_selection(
            relations=relations,
            in_dim=self.dec_in_dim,
            hidden_dim=self.dec_hidden_dim,
            out_dim=self.ntype_in_dim,
            num_heads=self.enc_heads,
            num_out_heads=self.dec_heads,
            num_layer=1,
            dropout=self.dropout,
            module_type=self.decoder_type,
            weight_T=self.weight_T,
            enc_dec="decoder",
        )
        self.criterion = partial(cosine_similarity, gamma=args.gamma)

    def initial_enc_dec_dim(self, relations):
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        self.edge_recons_encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)

    def forward(self, subgs, relations):
        try:
            node_feature_recons_loss = 0
            adjmatrix_recons_loss = 0
            ## Calculate node feature reconstruction loss
            if self.feat_recons:
                node_feature_recons_loss = self.mask_attribute_reconstruction(subgs[1], relations)
            if self.edge_recons:
                adjmatrix_recons_loss = self.mask_edge_reconstruction(subgs, relations)

            ## Calculate adjacent matrix reconstruction loss
            return node_feature_recons_loss + adjmatrix_recons_loss
        except RuntimeError as e:
            # Check if the error message contains CUDA-related information
            if "CUDA" in str(e):
                # Write the error message to a log file
                log_times = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
                with open(f"./log/error/gpu_error_{log_times}.log", "a") as f:
                    f.write(f"GPU Runtime Error: {e}\n")
                    f.write(traceback.format_exc())
            else:
                # If the error is not CUDA-related, print the error message
                print("RuntimeError:", e)

    def mask_edge_reconstruction(self, subgs, relations):
        ### src node should be encode to reconstruct adjacent matrix, rev relation should do same thing again
        ### we use dst-based graph's dst_node and src-based graph's dst_node to reconstruct adjacent matrix
        ### But if there are connection on same type of node, we just use dst_based_subg to reconstruct adjacent matrix
        src_based_subg = subgs[0]
        dst_based_subg = subgs[1]
        all_adjmatrix_recons_loss = 0.0
        ## this dict contain the dst src dec_rep pair to reconstruct the adjacent matrix
        all_rel_pair_dec_rep = {}

        dst_based_x = {
            "src_x": dst_based_subg.srcdata["feat"],
            "dst_x": dst_based_subg.dstdata["feat"],
        }
        src_based_x = {
            "src_x": src_based_subg.srcdata["feat"],
            "dst_x": src_based_subg.dstdata["feat"],
        }

        # print(dst_based_subg[("Gene", "Gene-Gene", "Gene")])
        for rel_tuple in relations:
            src_node, rel, dst_node = rel_tuple
            if not self.all_edge_recons and dst_node != self.target_type:
                continue
            rev_rel = f"{dst_node}-{src_node}"
            rev_tuple = (dst_node, rev_rel, src_node)
            mask_src_rels_subgraphs = src_based_subg[rev_rel]
            mask_dst_rels_subgraphs = dst_based_subg[rel]

            drop_edge = DropEdge(p=self.mask_rate)

            mask_src_rels_subgraphs = drop_edge(mask_src_rels_subgraphs)
            mask_dst_rels_subgraphs = drop_edge(mask_dst_rels_subgraphs)

            dst_enc_rep = self.encoder(
                {rel_tuple: mask_dst_rels_subgraphs},
                dst_node,
                dst_based_x["dst_x"][dst_node],
                {src_node: dst_based_x["src_x"][src_node]},
                "encoder",
                "adj_recons",
            )
            src_enc_rep = self.encoder(
                {rev_tuple: mask_src_rels_subgraphs},
                src_node,
                src_based_x["dst_x"][src_node],
                {dst_node: src_based_x["src_x"][dst_node]},
                "encoder",
                "adj_recons",
            )
            # src_ntype_rep=[self.edge_recons_encoder_to_decoder[ntype][idx](enc_rep) for idx,enc_rep in enumerate(src_ntype_enc_rep)]
            dst_rep = self.edge_recons_encoder_to_decoder(dst_enc_rep[rel])
            src_rep = self.edge_recons_encoder_to_decoder(src_enc_rep[rev_rel])

            origin_adj_matrix = dst_based_subg[rel].adj()

            origin_adj_tensor = origin_adj_matrix.to_dense()
            if src_node == dst_node:
                recon_adj_matrix = torch.mm(src_rep, src_rep.T)
            else:
                recon_adj_matrix = torch.mm(src_rep, dst_rep.T)
            rel_pair_loss = self.criterion(origin_adj_tensor, recon_adj_matrix)
            all_adjmatrix_recons_loss += rel_pair_loss
        return all_adjmatrix_recons_loss

    def mask_attribute_reconstruction(self, graph, relations):
        ### only used dst node to do attribute reconstruction
        ## x represent all node features
        ## use_x represent target predicted node's features
        src_x = graph.srcdata["feat"]
        dst_x = graph.dstdata["feat"]
        if not self.all_feat_recons:
            dst_x = {self.target_type: dst_x[self.target_type]}
        use_dst_x, (ntypes_mask_nodes, ntypes_keep_nodes) = self.encode_mask_noise(graph, dst_x)
        all_node_feature_recons_loss = 0.0
        for use_ntype, use_x in use_dst_x.items():
            rels_subgraphs = self.seperate_relation_graph(graph, relations, use_ntype)
            mask_nodes = ntypes_mask_nodes[use_ntype]
            if self.encoder_type == "SRN":
                enc_rep = self.encoder(rels_subgraphs, use_ntype, use_x, src_x, "encoder", "nf_recons")
            hidden_rep = self.encoder_to_decoder(enc_rep)
            # remask
            hidden_rep[mask_nodes] = 0.0
            # decoder module
            if self.decoder_type == "SRN":
                dec_rep = self.decoder(rels_subgraphs, use_ntype, hidden_rep, src_x, "decoder")
            elif self.decoder_type == "MLP":
                dec_rep = self.decoder(hidden_rep)
            ## calculate all type nodes feature reconstruction
            ntype_loss = self.criterion(dst_x[use_ntype][mask_nodes], dec_rep[mask_nodes])
            all_node_feature_recons_loss += ntype_loss
        return all_node_feature_recons_loss

    def seperate_relation_graph(self, graph, relations, use_ntype):
        rels_subgraphs = {}
        for rels_tuple in relations:
            ## dst_ntype include in relaions, then take its neighbor ntype as subgraph

            dst_ntype = rels_tuple[2]
            if use_ntype == dst_ntype:
                rels_subgraphs[rels_tuple] = graph[rels_tuple]
        return rels_subgraphs

    def encode_mask_noise(self, graph, ntype_x):
        ntypes_mask_x = {}
        ntypes_mask_nodes = {}
        ntypes_keep_nodes = {}
        for ntype, x in ntype_x.items():
            num_nodes = x.shape[0]
            # num_nodes = graph.num_nodes(ntype)

            permutation = torch.randperm(num_nodes)
            num_mask_nodes = int(self.mask_rate * num_nodes)
            mask_nodes = permutation[:num_mask_nodes]
            keep_nodes = permutation[num_mask_nodes:]
            # print(t)
            # print(ntype_x.shape)
            # print(torch.max(mask_nodes))
            mask_x = x.clone()
            mask_x[mask_nodes] = 0.0
            ntypes_mask_x[ntype] = mask_x
            ntypes_mask_nodes[ntype] = mask_nodes
            ntypes_keep_nodes[ntype] = keep_nodes
        return ntypes_mask_x, (ntypes_mask_nodes, ntypes_keep_nodes)
