import argparse
import dgl
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from utils.preprocess_ACM import ACMDataset
from utils.preprocess_HeCo import DBLPHeCoDataset
from utils.evaluate import score, LogisticRegression, MLP
from models.HGAE import HGAE
from models.HAN import HAN
from tqdm import tqdm

import numpy as np
heterogeneous_dataset = {
    'dblp': DBLPFourAreaDataset,
    'acm': ACMDataset,
    'heco_dblp': DBLPHeCoDataset
}
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def node_classification_evaluate(enc_feat, args, num_classes, labels, train_mask, val_mask, test_mask):
    print("---------Evaluation in Classification---------")
    eva_epoches = 200
    classifier = MLP(
        num_dim=args.num_hidden, num_classes=num_classes)
    classifier = classifier.to(device_1)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    enc_feat = enc_feat.to(device_1)
    emb = {"train": enc_feat[train_mask],
           "val": enc_feat[val_mask], "test": enc_feat[test_mask]}
    labels = {"train": labels[train_mask],
              "val": labels[val_mask], "test": labels[test_mask]}

    val_macro = []
    val_micro = []
    test_macro = []
    test_micro = []
    for epoch in tqdm(range(eva_epoches), position=0):
        classifier.train()
        train_output = classifier(emb["train"]).cpu()
        eva_loss = F.cross_entropy(
            train_output, labels["train"])
        optimizer.zero_grad()
        eva_loss.backward(retain_graph=True)
        optimizer.step()

        val_output = classifier(emb["val"]).cpu()
        val_micro_f1, val_macro_f1 = score(val_output, labels["val"])
        val_micro.append(val_micro_f1)
        val_macro.append(val_macro_f1)

    print(
        "\t Macro-F1:{:4f} Micro-F1:{:4f}".format(max(val_macro), max(val_micro)))


def train(args):
    data = heterogeneous_dataset[args.dataset]()
    metapaths = data.metapaths
    graph = data[0]
    hgs = [dgl.metapath_reachable_graph(graph, metapath).to(
        device_0) for metapath in metapaths]  # homogeneous graph divide by metapaths
    all_types = list(data._ntypes.values())
    ntype = data.predict_ntype
    num_classes = data.num_classes
    features = {t: graph.nodes[ntype].data['feat']
                for t in all_types if 'feat' in graph.nodes[t].data}

    ntype_labels = graph.nodes[ntype].data['label']
    train_mask = graph.nodes[ntype].data[f'train_mask_{args.ratio}']
    val_mask = graph.nodes[ntype].data[f'val_mask_{args.ratio}']
    test_mask = graph.nodes[ntype].data[f'test_mask_{args.ratio}']
    # model = HAN(num_metapaths=len(metapaths), in_dim=features[ntype].shape[1], hidden_dim=args.num_hidden, out_dim=num_classes,
    #            num_heads=args.num_heads, dropout=args.dropout)
    model = HGAE(num_metapath=len(metapaths),
                 in_dim=features[ntype].shape[1], args=args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    if (args.scheduler):
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=0.99)

        def scheduler(epoch): return (
            1 + np.cos((epoch) * np.pi / args.epoches)) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler)
    model = model.to(device_0)

    for epoch in range(args.epoches):
        model.train()
        features[ntype] = features[ntype].to(device_0)
        optimizer.zero_grad()
        loss = model(hgs, features[ntype])

        # output = model(hgs, features[ntype])
        # output = output.cpu()
        # loss = F.cross_entropy(output[train_mask], ntype_labels[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(
            f"Epoch: {epoch} Training Loss:{loss.item()} learning_rate={scheduler.get_last_lr()} ")

        if (epoch > 0 and epoch % 20 == 0):
            features[ntype] = features[ntype].to(device_0)
            enc_feat = model.encoder(hgs, features[ntype])
            node_classification_evaluate(
                enc_feat, args, num_classes, ntype_labels, train_mask, val_mask, test_mask)
        # with torch.no_grad():
        #     model.eval()


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
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--dropout', type=float,
                        default=0.4, help='dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--mask_rate', type=float,
                        default=0.5, help="masked node rates")
    parser.add_argument('--encoder', type=str, default="HAN",
                        help='heterogeneous encoder')
    parser.add_argument('--decoder', type=str, default="HAN",
                        help='Heterogeneous decoder')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--gamma', type=int,
                        default=3, help='gamma for cosine similarity')
    parser.add_argument('--scheduler', default=True,
                        help='scheduler for optimizer')
    args = parser.parse_args()

    train(args=args)
