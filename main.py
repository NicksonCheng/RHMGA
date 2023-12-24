import argparse
import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from utils.preprocess_ACM import ACMDataset
from utils.preprocess_HeCo import DBLPHeCoDataset
from utils.evaluate import score
from models.HGAE import HGAE, LogisticRegression
from models.HAN import HAN
from tqdm import tqdm

import numpy as np
heterogeneous_dataset = {
    'dblp': DBLPFourAreaDataset,
    'acm': ACMDataset,
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
    if (args.scheduler):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.99)
        # def scheduler(epoch): return (
        #     1 + np.cos((epoch) * np.pi / args.epoches)) * 0.5
        # # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=scheduler)
    model = model.to(device)

    for epoch in range(args.epoches):
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

        print(
            f"Epoch: {epoch} Training Loss:{loss.item()} learning_rate={scheduler.get_last_lr()} ")

        with torch.no_grad():
            model.eval()

    encoded_feature = model.encoder(hgs, ntype_features)
    classifier = LogisticRegression(
        num_dim=args.num_hidden, num_classes=num_classes)
    classifier = classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    print("---------Evaluation in Classification---------")
    for epoch in range(args.epoches):
        classifier.train()
        output = classifier(encoded_feature).cpu()
        eva_loss = F.cross_entropy(
            output[train_mask], ntype_labels[train_mask])
        optimizer.zero_grad()
        eva_loss.backward(retain_graph=True)
        optimizer.step()
        micro, macro = score(output[val_mask], ntype_labels[val_mask])

        print(
            f"Epoch: {epoch} micro_f1:{micro} macro_f1:{macro} learning_rate={scheduler.get_last_lr()} ")


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
    parser.add_argument('--scheduler', default=True,
                        help='scheduler for optimizer')
    args = parser.parse_args()

    train(args=args)
