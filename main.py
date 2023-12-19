import argparse
import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from utils.preprocess_HeCo import DBLPHeCoDataset
from models.HGAE import HGAE
from models.HAN import HAN
from tqdm import tqdm
heterogeneous_dataset = {
    'dblp': DBLPFourAreaDataset,
    'heco_dblp': DBLPHeCoDataset
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    data = heterogeneous_dataset[args.dataset]()
    metapaths = data.metapaths
    graph = data[0]
    hgs = [dgl.metapath_reachable_graph(graph, metapath).to(
        device) for metapath in metapaths]  # homogeneous graph divide by metapaths
    ntype = data.predict_ntype
    num_classes = data.num_classes
    ntype_features = graph.nodes[ntype].data['feat']
    ntype_labels = graph.nodes[ntype].data['label']
    train_mask = graph.nodes[ntype].data[f'train_mask_{args.ratio}']
    val_mask = graph.nodes[ntype].data[f'val_mask_{args.ratio}']
    test_mask = graph.nodes[ntype].data[f'test_mask_{args.ratio}']
    # model = HAN(num_metapaths=len(metapaths), in_dim=ntype_features.shape[1], hidden_dim=args.num_hidden, out_dim=num_classes,
    #            num_heads=args.num_heads, dropout=args.dropout)
    model = HGAE(num_metapath=len(metapaths),
                 in_dim=ntype_features.shape[1], args=args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.99)
    model = model.to(device)

    for epoch in tqdm(range(args.epoches)):
        model.train()
        ntype_features = ntype_features.to(device)
        optimizer.zero_grad()
        loss = model(hgs, ntype_features)
        # output = model(hgs, ntype_features)
        # output = output.cpu()
        # loss = F.cross_entropy(output[train_mask], ntype_labels[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch: {epoch} Training Loss:{loss.item()}")

        with torch.no_grad():
            model.eval()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Heterogeneous Project")
    parser.add_argument("--dataset", type=str, default="dblp")
    parser.add_argument("--ratio", type=int, default=20)
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--num_layer", type=int, default=3,
                        help="number of model layer")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of attention output heads")
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='number of hidden units')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--mask_rate', type=float,
                        default=0.5, help="masked node rates")
    parser.add_argument('--encoder', type=str, default="HAN",
                        help='heterogeneous encoder')
    parser.add_argument('--decoder', type=str, default="HAN",
                        help='Heterogeneous decoder')
    parser.add_argument('--weight-decay', type=float,
                        default=2e-4, help='weight decay')
    parser.add_argument('--gamma', type=int,
                        default=3, help='gamma for cosine similarity')
    args = parser.parse_args()

    train(args=args)
