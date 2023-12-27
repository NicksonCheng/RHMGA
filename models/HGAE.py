import torch
import torch_geometric
import torch.nn as nn
from models.HAN import HAN
from utils.evaluate import cosine_similarity, mse
from functools import partial


def module_selection(num_m, in_dim, hidden_dim, out_dim, num_heads, num_out_heads, num_layer, dropout, module_type):
    if (module_type == "HAN"):
        return HAN(num_metapaths=num_m,
                   in_dim=in_dim,
                   hidden_dim=hidden_dim,
                   out_dim=out_dim,
                   num_layer=num_layer,
                   num_heads=num_heads,
                   num_out_heads=num_out_heads,
                   dropout=dropout)


class HGAE(nn.Module):
    def __init__(self,
                 num_metapath,
                 in_dim,
                 args
                 ):
        super(HGAE, self).__init__()
        self.mask_rate = args.mask_rate
        self.in_dim = in_dim
        self.hidden_dim = args.num_hidden
        self.num_layer = args.num_layer
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.dropout = args.dropout
        self.gamma = args.gamma

        # encoder/decoder hidden dimension
        self.enc_dim = self.hidden_dim//self.num_heads
        self.enc_heads = self.num_heads

        self.dec_in_dim = self.hidden_dim  # enc_dim * enc_heads
        self.dec_hidden_dim = self.hidden_dim//self.num_heads
        self.dec_heads = self.num_out_heads

        self.encoder = module_selection(
            num_m=num_metapath,
            in_dim=in_dim,
            hidden_dim=self.enc_dim,
            out_dim=self.enc_dim,
            num_heads=self.enc_heads,
            num_out_heads=self.enc_heads,
            num_layer=self.num_layer,
            dropout=self.dropout,
            module_type=self.encoder_type,
        )
        # linear transformation from encoder to decoder
        self.encoder_to_decoder = nn.Linear(
            self.dec_in_dim, self.dec_in_dim, bias=False)

        self.decoder = module_selection(
            num_m=num_metapath,
            in_dim=self.dec_in_dim,
            hidden_dim=self.dec_hidden_dim,
            out_dim=self.in_dim,
            num_heads=self.enc_heads,
            num_out_heads=self.dec_heads,
            num_layer=1,
            dropout=self.dropout,
            module_type=self.decoder_type,
        )

        self.criterion = partial(cosine_similarity, gamma=args.gamma)

    def forward(self, hgs, x):
        predicted_x, mask_nodes = self.mask_attribute_prediction(
            hgs, x)
        loss = self.criterion(x[mask_nodes], predicted_x[mask_nodes])
        return loss

    def mask_attribute_prediction(self, hgs, x):
        use_x, (mask_nodes, keep_nodes) = self.encode_mask_noise(hgs, x)
        enc_rep = self.encoder(hgs, use_x)
        rep = self.encoder_to_decoder(enc_rep)
        # remask
        rep[mask_nodes] = 0.0

        dec_rep = self.decoder(hgs, rep)

        # print(x[mask_nodes])
        # print(dec_rep[mask_nodes])

        return dec_rep, mask_nodes

    def encode_mask_noise(self, hgs, x):
        num_nodes = hgs[0].num_nodes()
        permutation = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_rate*num_nodes)
        mask_nodes = permutation[:num_mask_nodes]
        keep_nodes = permutation[num_mask_nodes:]

        mask_x = x.clone()
        mask_x[mask_nodes] = 0.0
        return mask_x, (mask_nodes, keep_nodes)
