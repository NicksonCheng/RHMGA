import torch
import torch_geometric
import torch.nn as nn
from models.HAN import HAN
from models.SRN import Schema_Relation_Network
from models.HAN_SRN import Metapath_Relation_Network
from utils.evaluate import cosine_similarity, mse
from functools import partial
import traceback


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
    num_relations,
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
            num_relations=num_relations,
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
    def __init__(self, num_metapath, num_relations, all_types, target_in_dim, ntype_in_dim, args):
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
            num_relations=num_relations,
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
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)

        self.decoder = module_selection(
            num_m=num_metapath,
            num_relations=num_relations,
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

    def forward(self, graph, relations, mp_subgraphs, src_x, dst_x):
        try:
            # Code that may raise a CUDA runtime error
            predicted_x, ntypes_mask_nodes = self.mask_attribute_prediction(
                graph, relations, mp_subgraphs, src_x, dst_x
            )
            all_type_reconstruction_loss = 0.0
            for t in self.all_types:
                # print(predicted_x[t].shape)
                mask_nodes = ntypes_mask_nodes[t]
                ntype_loss = self.criterion(dst_x[t][mask_nodes], predicted_x[t][mask_nodes])
                # print(f"{t}:{ntype_loss}")
                all_type_reconstruction_loss += ntype_loss
            return all_type_reconstruction_loss
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

    def mask_edge_prediction(self, x):
        pass

    def mask_attribute_prediction(self, graph, relations, mp_subgraphs, src_x, dst_x):
        # x represent all node features
        # use_x represent target predicted node's features
        use_dst_x, (ntypes_mask_nodes, ntypes_keep_nodes) = self.encode_mask_noise(graph, dst_x)
        ntypes_dec_rep = {}
        for use_ntype, use_x in use_dst_x.items():
            mask_nodes = ntypes_mask_nodes[use_ntype]
            if self.encoder_type == "HAN":
                enc_rep = self.encoder(mp_subgraphs, use_x)
            elif self.encoder_type == "SRN":
                enc_rep = self.encoder(graph, relations, use_ntype, use_x, src_x, "encoder")
            hidden_rep = self.encoder_to_decoder(enc_rep)
            # remask
            hidden_rep[mask_nodes] = 0.0
            # decoder module
            if self.decoder_type == "HAN":
                dec_rep = self.decoder(mp_subgraphs, hidden_rep)
            elif self.decoder_type == "SRN":
                dec_rep = self.decoder(graph, relations, use_ntype, hidden_rep, src_x, "decoder")
            elif self.decoder_type == "MLP":
                dec_rep = self.decoder(hidden_rep)
            ntypes_dec_rep[use_ntype] = dec_rep
        ## return all type nodes representation
        return ntypes_dec_rep, ntypes_mask_nodes

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
