import torch
import torch_geometric
import torch.nn as nn
from models.HAN import HAN
from models.SRN import Schema_Relation_Network
from models.HAN_SRN import Metapath_Relation_Network
from utils.evaluate import cosine_similarity, mse
from functools import partial
from dgl import DropEdge
import traceback
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
    num_m,
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
            num_metapaths=num_m,
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
    def __init__(self, num_metapath, relations, all_types, target_in_dim, ntype_in_dim, args):
        super(HGARME, self).__init__()
        self.mask_rate = args.mask_rate
        self.target_in_dim = target_in_dim
        self.ntype_in_dim = ntype_in_dim
        self.hidden_dim = args.num_hidden
        self.num_layer = args.num_layer
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.dropout = args.dropout
        self.gamma = args.gamma
        self.all_types = all_types

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
        self.weight_T = nn.ModuleDict(
            {t: nn.Linear(self.ntype_in_dim[t], self.hidden_dim) for t in self.all_types}
        )
        self.encoder = module_selection(
            num_m=num_metapath,
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
            num_m=num_metapath,
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
        self.edge_recons_encoder_to_decoder = nn.Linear(
            self.dec_in_dim, self.dec_in_dim, bias=False
        )
        # ntype_dst_enc_dec={}

        # for ntype in self.all_types:
        #     ntype_dst_neighbors=nn.ModuleList([
        #         nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        #         for _ in range(len(relations))
        #     ])
        #     ntype_dst_enc_dec[ntype]=ntype_dst_neighbors

        # self.edge_recons_encoder_to_decoder = ntype_dst_enc_dec

    def forward(self, graph, relations, mp_subgraphs, src_x, dst_x):
        try:
            ## Code that may raise a CUDA runtime error
            ## calculate node feature reconstruction loss

            # node_feature_recons_loss = self.mask_attribute_reconstruction(
            #     graph, relations, mp_subgraphs, src_x, dst_x
            # )

            adjmatrix_recons_loss = self.mask_edge_reconstruction(graph, relations, src_x, dst_x)
            ## calculate adjacent matrix reconstruction loss

            return node_feature_recons_loss + adjmatrix_recons_loss
        except RuntimeError as e:
            # Check if the error message contains CUDA-related information
            if "CUDA" in str(e):
                # Write the error message to a log file
                with open("./log/gpu_error.log", "a") as f:
                    f.write(f"GPU Runtime Error: {e}\n")
                    f.write(traceback.format_exc())
            else:
                # If the error is not CUDA-related, print the error message
                print("RuntimeError:", e)

    def mask_edge_reconstruction(self, graph, relations, src_x, dst_x):
        all_adjmatrix_recons_loss = 0.0
        all_rel_dec_rep = {}

        for ntype, x in dst_x.items():
            mask_rels_subgraphs = self.seperate_relation_graph(graph, relations, ntype)
            drop_edge = DropEdge(p=self.mask_rate)
            for rel in mask_rels_subgraphs.keys():
                mask_rels_subgraphs[rel]["graph"] = drop_edge(mask_rels_subgraphs[rel]["graph"])
                # src_ntype_enc_rep = self.encoder(masked_subgs, relations, ntype, x, src_x, "encoder","adj_recons")
            enc_rep = self.encoder(
                mask_rels_subgraphs, relations, ntype, x, src_x, "encoder", "adj_recons"
            )
            # src_ntype_rep=[self.edge_recons_encoder_to_decoder[ntype][idx](enc_rep) for idx,enc_rep in enumerate(src_ntype_enc_rep)]

            rep = self.edge_recons_encoder_to_decoder(enc_rep)

            # src_ntype_dec_rep=self.decoder(masked_subgs, relations, ntype, src_ntype_rep, src_x, "decoder")
            src_ntype_dec_rep = self.decoder(
                mask_rels_subgraphs, relations, ntype, rep, src_x, "decoder", "adj_recons"
            )
            for rel, dec_rep in src_ntype_dec_rep.items():
                all_rel_dec_rep[rel] = dec_rep
        for rel, dec_rep in all_rel_dec_rep.items():
            print(rel, dec_rep.shape)
        adj_matrix = dgl.to_scipy_sparse_matrix(graph)
        adj_tensor = torch.tensor(adj_matrix.todense())
        print(adj_tensor.shape)
        exit()
        return all_adjmatrix_recons_loss

    def mask_attribute_reconstruction(self, graph, relations, mp_subgraphs, src_x, dst_x):
        # x represent all node features
        # use_x represent target predicted node's features
        use_dst_x, (ntypes_mask_nodes, ntypes_keep_nodes) = self.encode_mask_noise(graph, dst_x)

        all_node_feature_recons_loss = 0.0
        for use_ntype, use_x in use_dst_x.items():
            rels_subgraphs = self.seperate_relation_graph(graph, relations, use_ntype)
            mask_nodes = ntypes_mask_nodes[use_ntype]
            if self.encoder_type == "HAN":
                enc_rep = self.encoder(mp_subgraphs, use_x)
            elif self.encoder_type == "SRN":
                enc_rep = self.encoder(
                    rels_subgraphs, relations, use_ntype, use_x, src_x, "encoder", "nf_recons"
                )
            hidden_rep = self.encoder_to_decoder(enc_rep)
            # remask
            hidden_rep[mask_nodes] = 0.0
            # decoder module
            if self.decoder_type == "HAN":
                dec_rep = self.decoder(mp_subgraphs, hidden_rep)
            elif self.decoder_type == "SRN":
                dec_rep = self.decoder(
                    rels_subgraphs, relations, use_ntype, hidden_rep, src_x, "decoder"
                )
            elif self.decoder_type == "MLP":
                dec_rep = self.decoder(hidden_rep)
            ## calculate all type nodes feature reconstruction
            ntype_loss = self.criterion(dst_x[use_ntype][mask_nodes], dec_rep[mask_nodes])
            all_node_feature_recons_loss += ntype_loss
        return all_node_feature_recons_loss

    def seperate_relation_graph(self, graph, relations, dst_ntype, is_mask_edge=False):
        rels_subgraphs = {}

        for rels_tuple in relations:
            ## dst_ntype include in relaions, then take its neighbor ntype as subgraph
            if dst_ntype not in rels_tuple:
                continue
            # reverse the relation to make dst_ntype as dst_node in graph

            reverse_rels_tuple = list(rels_tuple)
            reverse_rels_tuple[1] = rels_tuple[1][::-1]  # (relation)^-1
            reverse_rels_tuple = tuple(reversed(reverse_rels_tuple))

            ## if the target_ntype is src_type in relation, we need to reverse the relation
            if dst_ntype == rels_tuple[0]:
                reverse_src_ntype, reverse_rel, reverse_dst_ntype = reverse_rels_tuple
                rels_subgraphs[reverse_rel] = {}
                rels_subgraphs[reverse_rel]["src_ntype"] = reverse_src_ntype
                rels_subgraphs[reverse_rel]["dst_ntype"] = reverse_dst_ntype
                rels_subgraphs[reverse_rel]["original_rel"] = rels_tuple[1]
                rels_subgraphs[reverse_rel]["graph"] = graph[reverse_rel].clone()
            else:
                src_ntype, rel, dst_ntype = rels_tuple
                rels_subgraphs[rel] = {}
                rels_subgraphs[rel]["src_ntype"] = src_ntype
                rels_subgraphs[rel]["dst_ntype"] = dst_ntype
                rels_subgraphs[rel]["original_rel"] = rels_tuple[1]
                rels_subgraphs[rel]["graph"] = graph[rel].clone()

        return rels_subgraphs

    def encode_mask_noise(self, graph, dst_x):
        ntypes_mask_x = {}
        ntypes_mask_nodes = {}
        ntypes_keep_nodes = {}
        for ntype, x in dst_x.items():
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
