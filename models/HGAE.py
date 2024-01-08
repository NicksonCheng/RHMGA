import torch
import torch_geometric
import torch.nn as nn
from models.HAN import HAN
from models.SRN import Schema_Relation_Network
from models.HAN_SRN import Metapath_Relation_Network
from utils.evaluate import cosine_similarity, mse
from functools import partial

class MultiLayerPerception(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerPerception, self).__init__()
        self.hidden = input_dim*2
        self.fc1 = nn.Linear(input_dim, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def module_selection(num_m, num_relations, in_dim, hidden_dim, out_dim,
                     num_heads, num_out_heads, num_layer, dropout, module_type, weight_T):
    if (module_type == "HAN"):
            return HAN(num_metapaths=num_m,
                   in_dim=in_dim,
                   hidden_dim=hidden_dim,
                   out_dim=out_dim,
                   num_layer=num_layer,
                   num_heads=num_heads,
                   num_out_heads=num_out_heads,
                   dropout=dropout)
    elif (module_type == "SRN"):
        return Schema_Relation_Network(num_relations=num_relations,
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim,
                                       num_layer=num_layer,
                                       num_heads=num_heads,
                                       num_out_heads=num_out_heads,
                                       dropout=dropout,
                                       weight_T=weight_T)
    elif(module_type=="HAN_SRN"):
        return Metapath_Relation_Network(num_metapaths=num_m,
                                         num_relations=num_relations,
                                         hidden_dim=hidden_dim,
                                         out_dim=out_dim,
                                         num_han_layer=num_layer,
                                         num_srn_layer=num_layer,
                                         num_heads=num_heads,
                                         num_out_heads=num_out_heads,
                                         dropout=dropout,
                                         weight_T=weight_T)

    elif(module_type=="MLP"):
        return MultiLayerPerception(input_dim=hidden_dim,
                                     output_dim=out_dim)



class HGAE(nn.Module):
    def __init__(self,
                 num_metapath,
                 num_relations,
                 target_in_dim,
                 all_in_dim,
                 args
                 ):
        super(HGAE, self).__init__()
        self.mask_rate = args.mask_rate
        self.target_in_dim = target_in_dim
        self.all_in_dim = all_in_dim
        self.hidden_dim = args.num_hidden
        self.num_layer = args.num_layer
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.dropout = args.dropout
        self.gamma = args.gamma

        # encoder/decoder hidden dimension
        if (self.encoder_type == "HAN"):
            self.enc_dim = self.hidden_dim//self.num_heads
        else:
            self.enc_dim = self.hidden_dim
        self.enc_heads = self.num_heads

        self.dec_in_dim = self.hidden_dim  # enc_dim * enc_heads
        self.dec_hidden_dim = self.hidden_dim//self.num_heads
        self.dec_heads = self.num_out_heads

        ## project all type of node into same dimension
        self.weight_T = nn.ModuleList([
            nn.Linear(dim, self.hidden_dim)
            for dim in self.all_in_dim
        ])

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
            weight_T=self.weight_T
        )
        # linear transformation from encoder to decoder
        self.encoder_to_decoder = nn.Linear(
            self.dec_in_dim, self.dec_in_dim, bias=False)

        self.decoder = module_selection(
            num_m=num_metapath,
            num_relations=num_relations,
            in_dim=self.dec_in_dim,
            hidden_dim=self.dec_hidden_dim,
            out_dim=self.target_in_dim,
            num_heads=self.enc_heads,
            num_out_heads=self.dec_heads,
            num_layer=1,
            dropout=self.dropout,
            module_type=self.decoder_type,
            weight_T=self.weight_T
        )
        self.criterion = partial(cosine_similarity, gamma=args.gamma)

    def forward(self, mp_subgraphs, sc_subgraphs, x, ntype):

        predicted_x, mask_nodes = self.mask_attribute_prediction(
            mp_subgraphs, sc_subgraphs, x, ntype)
        loss = self.criterion(x[ntype][mask_nodes], predicted_x[mask_nodes])
        return loss

    def mask_edge_prediction(self, mp_subgraphs, x):
        pass

    def mask_attribute_prediction(self, mp_subgraphs, sc_subgraphs, x, ntype):
        # x represent all node features
        # use_x represent target predicted node's features
        use_x, (mask_nodes, keep_nodes) = self.encode_mask_noise(
            mp_subgraphs, x[ntype])
        if (self.encoder_type == "HAN"):
            enc_rep = self.encoder(mp_subgraphs, use_x)
        else:
            enc_rep = self.encoder(sc_subgraphs, use_x, x)

        hidden_rep = self.encoder_to_decoder(enc_rep)
        # remask
        hidden_rep[mask_nodes] = 0.0

        if (self.decoder_type == "HAN"):
            dec_rep = self.decoder(mp_subgraphs, hidden_rep)
        else:
            dec_rep = self.decoder(sc_subgraphs, hidden_rep, x)
        print(dec_rep.shape)
        exit()
        # print(x[mask_nodes])
        # print(dec_rep[mask_nodes])

        return dec_rep, mask_nodes

    def encode_mask_noise(self, mp_subgraphs, ntype_x):
        num_nodes = mp_subgraphs[0].num_nodes()
        permutation = torch.randperm(num_nodes, device=ntype_x.device)
        num_mask_nodes = int(self.mask_rate*num_nodes)
        mask_nodes = permutation[:num_mask_nodes]
        keep_nodes = permutation[num_mask_nodes:]

        mask_x = ntype_x.clone()
        mask_x[mask_nodes] = 0.0
        return mask_x, (mask_nodes, keep_nodes)
