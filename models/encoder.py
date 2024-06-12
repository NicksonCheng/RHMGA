import torch
import torch.nn as nn
from models.HAN import HAN
from models.SRN import Schema_Relation_Network
from models.HGraphSAGE import HGraphSAGE
from utils.evaluate import cosine_similarity, mse
from functools import partial
from dgl.transforms import DropEdge
from collections import Counter
import traceback
import datetime
import dgl


class HGARME_E(nn.Module):
    def __init__(self, relations, target_type, all_types, target_in_dim, ntype_in_dim, args):
        super(HGARME_E, self).__init__()
        self.target_in_dim = target_in_dim
        self.ntype_in_dim = ntype_in_dim
        self.target_type = target_type
        self.all_types = all_types
        # self.mask_rate = args.mask_rate
        self.mask_rate = args.dynamic_mask_rate
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
        self.ntype_enc_mask_token = {t: nn.Parameter(torch.zeros(1, self.ntype_in_dim[t])) for t in self.all_types}

        self.encoder = HGraphSAGE(
            relations=relations,
            hidden_dim=self.enc_dim,
            ntype_out_dim=self.enc_dim,
            num_layer=self.num_layer,
            weight_T=self.weight_T,
            status="encoder",
        )
        # linear transformation from encoder to decoder
        self.initial_status_dim(relations=relations)
        self.criterion = partial(cosine_similarity, gamma=args.gamma)

    def initial_status_dim(self, relations):
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        self.edge_recons_encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)

    def forward(self, subgs, relations, recons_type, epoch=None):
        try:
            ## Calculate node feature reconstruction loss
            if recons_type == "feature":
                return self.mask_attribute_reconstruction(subgs, relations, epoch)
            else:
                return self.mask_edge_reconstruction(subgs, relations, epoch)
        except RuntimeError as e:
            # Check if the error message contains CUDA-related information
            if "CUDA" in str(e):
                # Write the error messagse to a log file
                log_times = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
                with open(f"./log/error/gpu_error_{log_times}.log", "a") as f:
                    f.write(f"GPU Runtime Error: {e}\n")
                    f.write(traceback.format_exc())
            else:
                # If the error is not CUDA-related, print the error message
                print("RuntimeError:", e)

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split("~")]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(",")]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def mask_edge_reconstruction(self, subgs, relations, epoch=None):
        ### src node should be encode to reconstruct adjacent matrix, rev relation should do same thing again
        ### we use dst-based graph's dst_node and src-based graph's dst_node to reconstruct adjacent matrix
        ### But if there are connection on same type of node, we just use dst_based_subg to reconstruct adjacent matrix
        curr_mask_rate = self.get_mask_rate(self.mask_rate, epoch=epoch)

        src_based_subg = subgs[-2]
        dst_based_subg = subgs[-1]
        all_adjmatrix_recons_loss = 0.0
        ## this dict contain the dst src dec_rep pair to reconstruct the adjacent matrix
        all_rel_pair_dec_rep = {}

        dst_based_x = dst_based_subg.dstdata["feat"]
        src_based_x = src_based_subg.dstdata["feat"]
        dst_based_embs = {}
        src_based_embs = {}
        num_loss = 0
        # print(dst_based_subg[("Gene", "Gene-Gene", "Gene")])
        for use_ntype, use_x in dst_based_x.items():
            if not self.all_edge_recons and use_ntype != self.target_type:
                continue

            dst_enc_rep, att_mp = self.encoder(subgs, 1, relations, use_ntype, use_x, "edge_recons_encoder", curr_mask_rate=curr_mask_rate)
            # src_ntype_rep=[self.edge_recons_encoder_to_decoder[ntype][idx](enc_rep) for idx,enc_rep in enumerate(src_ntype_enc_rep)]
            dst_based_embs[use_ntype] = self.encoder_to_decoder(dst_enc_rep)
        for use_ntype, use_x in src_based_x.items():
            if not self.all_edge_recons and use_ntype != self.target_type:
                continue

            src_enc_rep, att_mp = self.encoder(subgs, 2, relations, use_ntype, use_x, "edge_recons_encoder", curr_mask_rate=curr_mask_rate)
            # src_ntype_rep=[self.edge_recons_encoder_to_decoder[ntype][idx](enc_rep) for idx,enc_rep in enumerate(src_ntype_enc_rep)]

            src_based_embs[use_ntype] = self.encoder_to_decoder(src_enc_rep)
        for rel_tuple in relations:
            src_node, rel, dst_node = rel_tuple
            origin_adj_matrix = dst_based_subg[rel].adj()

            origin_adj_tensor = origin_adj_matrix.to_dense()
            if src_node == dst_node:
                src_dec_rep = src_based_embs[dst_node]
                recon_adj_matrix = torch.mm(src_dec_rep, src_dec_rep.T)
            else:
                src_dec_rep = src_based_embs[src_node]
                dst_dec_rep = dst_based_embs[dst_node]
                recon_adj_matrix = torch.mm(src_dec_rep, dst_dec_rep.T)
            rel_pair_loss = self.criterion(origin_adj_tensor, recon_adj_matrix)
            all_adjmatrix_recons_loss += rel_pair_loss
            num_loss += 1
        return all_adjmatrix_recons_loss / num_loss

    def mask_attribute_reconstruction(self, subgs, relations, epoch=None):
        ### only used dst node to do attribute reconstruction
        ## x represent all node features
        ## use_x represent target predicted node's features

        curr_mask_rate = self.get_mask_rate(self.mask_rate, epoch=epoch)

        dst_x = subgs[-1].dstdata["feat"]
        if not self.all_feat_recons:
            dst_x = {self.target_type: dst_x[self.target_type]}
        use_dst_x, (ntypes_mask_nodes, ntypes_keep_nodes) = self.encode_mask_noise(subgs, dst_x, curr_mask_rate)
        all_node_feature_recons_loss = 0.0
        ntype_hidden_rep = {}
        for use_ntype, use_x in use_dst_x.items():
            mask_nodes = ntypes_mask_nodes[use_ntype]
            enc_rep, att_mp = self.encoder(subgs, 1, relations, use_ntype, use_x, "encoder")

            hidden_rep = self.encoder_to_decoder(enc_rep)

            ntype_hidden_rep[use_ntype] = hidden_rep

        return ntype_hidden_rep, ntypes_mask_nodes

    def encode_mask_noise(self, graph, ntype_x, mask_rate):
        ntypes_mask_x = {}
        ntypes_mask_nodes = {}
        ntypes_keep_nodes = {}
        for ntype, x in ntype_x.items():
            num_nodes = x.shape[0]
            # num_nodes = graph.num_nodes(ntype)

            permutation = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = permutation[:num_mask_nodes]
            keep_nodes = permutation[num_mask_nodes:]
            #### tmp block
            replace_rate = 0.3
            leave_unchanged = 0.2
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            num_leave_nodes = int(leave_unchanged * num_mask_nodes)
            num_noise_nodes = int(replace_rate * num_mask_nodes)
            num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
            token_nodes = mask_nodes[perm_mask[:num_real_mask_nodes]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            enc_mask_token = self.ntype_enc_mask_token[ntype].to(x.device)
            out_x[token_nodes] += enc_mask_token
            if num_noise_nodes > 0:
                out_x[noise_nodes] = x[noise_to_be_chosen]
            ntypes_mask_x[ntype] = out_x
            ntypes_mask_nodes[ntype] = mask_nodes
            ntypes_keep_nodes[ntype] = keep_nodes

            ### tmp block

            # mask_x = x.clone()
            # mask_x[mask_nodes] = 0.0
            # ntypes_mask_x[ntype] = mask_x
            # ntypes_mask_nodes[ntype] = mask_nodes
            # ntypes_keep_nodes[ntype] = keep_nodes
        return ntypes_mask_x, (ntypes_mask_nodes, ntypes_keep_nodes)
