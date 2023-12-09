import argparse
import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F
from preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from models.HAN import HAN
from tqdm import tqdm
from utils import f1_score
heterogeneous_dataset = {
    'dblp': DBLPFourAreaDataset
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    data = heterogeneous_dataset[args.dataset]()
    metapaths = data.metapaths
    graph = data[0]
    subgraph = [dgl.metapath_reachable_graph(graph, metapath).to(
        device) for metapath in metapaths]  # homogeneous graph divide by metapaths
    ntype = data.predict_ntype
    num_classes = data.num_classes
    features = graph.nodes[ntype].data['feat']
    labels = graph.nodes[ntype].data['label']
    train_mask = graph.nodes[ntype].data['train_mask']
    val_mask = graph.nodes[ntype].data['val_mask']
    test_mask = graph.nodes[ntype].data['test_mask']
    model = HAN(num_metapaths=len(metapaths), in_dim=features.shape[1], hidden_dim=args.num_hidden, out_dim=num_classes,
                num_heads=args.num_heads, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    model = model.to(device)
    score = f1_score
    for epoch in tqdm(range(args.epoches)):
        model.train()
        features = features.to(device)
        output = model(subgraph, features).cpu()
        loss = F.cross_entropy(output[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch} Training Loss:{loss}")

        with optimizer.no_grad():
            model.eval()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Heterogeneous Project")
    parser.add_argument("--dataset", type=str, default="dblp")
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of attention heads")
    parser.add_argument('--num_hidden', type=int, default=16,
                        help='number of hidden units')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout probability')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=0.001, help='weight decay')
    args = parser.parse_args()

    train(args=args)
