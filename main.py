from datetime import datetime
import time
import os
import sys
import argparse

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from utils.preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from utils.preprocess_ACM import ACMDataset
from utils.preprocess_HeCo import (
    DBLPHeCoDataset,
    ACMHeCoDataset,
    AMinerHeCoDataset,
    FreebaseHeCoDataset,
)
from utils.preprocess_PubMed import PubMedDataset
from utils.evaluate import score, LogisticRegression, MLP, node_classification_evaluate
from utils.utils import load_config, colorize, name_file
from models.HGARME import HGARME
from models.HAN import HAN
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
heterogeneous_dataset = {
    # "dblp": DBLPFourAreaDataset,
    # "acm": ACMDataset,
    # "heco_acm": {"name": ACMHeCoDataset, "relations": [("author", "ap", "paper")]},
    # "heco_dblp": {
    #     "name": DBLPHeCoDataset,
    #     # 'relations': [('paper', 'pa', 'author'), ('paper', 'pt', 'term'),(('paper', 'pc', 'conference'))]
    #     "relations": [
    #         ("paper", "pa", "author"),
    #     ],
    # },
    "heco_freebase": {
        "name": FreebaseHeCoDataset,
    },
    "PubMed": {
        "name": PubMedDataset,
    },
    # "heco_aminer": {
    #     "name": AMinerHeCoDataset,
    #     "relations": [("paper", "pa", "author"), ("paper", "pr", "reference")],
    # },
}


def train(args):
    start_t = time.time()
    # torch.cuda.set_device()
    device_0 = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    device_1 = torch.device(f"cuda:{args.devices ^ 1 }" if torch.cuda.is_available() else "cpu")

    data = heterogeneous_dataset[args.dataset]["name"]()
    print("Preprocessing Time taken:", time.time() - start_t, "seconds")
    start_t = time.time()
    relations = data.relations
    graph = data[0].to(device_0)

    all_types = list(data._ntypes.values())
    target_type = data.predict_ntype
    target_type_labels = graph.nodes[target_type].data["label"]
    num_classes = data.num_classes
    features = {
        t: graph.nodes[t].data["feat"].to(device_0)
        for t in all_types
        if "feat" in graph.nodes[t].data
    }
    train_nids = {ntype: torch.arange(graph.num_nodes(ntype)).to(device_0) for ntype in all_types}

    masked_graph = data.masked_graph

    # sampler = MultiLayerFullNeighborSampler(2)
    sampler = MultiLayerNeighborSampler([10, 5])
    dataloader = DataLoader(
        graph,
        train_nids,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        use_uva=False,
    )
    model = HGARME(
        relations=relations,
        target_type=target_type,
        all_types=all_types,
        target_in_dim=features[target_type].shape[1],
        ntype_in_dim={ntype: feat.shape[1] for ntype, feat in features.items()},
        args=args,
    )
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     # model = nn.DataParallel(model)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.devices], output_device=args.devices
    #     )
    model = model.to(device_0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler:
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=0.99)

        def scheduler(epoch):
            return (1 + np.cos((epoch) * np.pi / args.epoches)) * 0.5

        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    performance = {}
    total_loss = []
    print("Modeling Time taken:", time.time() - start_t, "seconds")
    log_times = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
    for epoch in tqdm(
        range(args.epoches), total=args.epoches, desc=colorize("Epoch Training", "blue")
    ):
        model.train()
        train_loss = 0.0
        for i, mini_batch in enumerate(dataloader):
            src_nodes, dst_nodes, subgs = mini_batch
            # print("----------------------------------")
            # for subg in subgs:
            #     print(subg)
            # print("----------------------------------")
            loss = model(subgs, relations)
            train_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(f"Batch {i} Loss: {loss.item()}")
        avg_train_loss = train_loss / len(dataloader)
        print(
            f"Epoch:{epoch+1}/{args.epoches} Training Loss:{(avg_train_loss)} Learning_rate={scheduler.get_last_lr()}"
        )
        total_loss.append(avg_train_loss)
        ## Evaluate Embedding Performance
        if epoch > 0 and ((epoch + 1) % args.eva_interval) == 0:
            # if True:
            file_name = name_file(args, "log", log_times)
            with open(file_name, "a") as log_file:

                log_file.write(f"Epoches:{epoch}-----------------------------------\n")

                rels_subgraphs = model.seperate_relation_graph(graph, relations, target_type)
                enc_feat = model.encoder(
                    rels_subgraphs,
                    target_type,
                    features[target_type],
                    features,
                    "encoder",
                )
                if data.has_label_ratio:
                    for ratio in data.label_ratio:
                        max_acc, max_micro, max_macro = node_classification_evaluate(
                            device_1,
                            enc_feat,
                            args,
                            num_classes,
                            target_type_labels,
                            masked_graph[ratio],
                        )
                        if ratio not in performance:
                            performance[ratio] = {"Acc": [], "Micro-F1": [], "Macro-F1": []}

                        performance[ratio]["Acc"].append(max_acc)
                        performance[ratio]["Micro-F1"].append(max_micro)
                        performance[ratio]["Macro-F1"].append(max_macro)
                        log_file.write(
                            "\t Label Rate:{}% [Accuracy:{:4f} Micro-F1:{:4f} Macro-F1:{:4f}  ]\n".format(
                                ratio,
                                max_acc,
                                max_micro,
                                max_macro,
                            )
                        )

                else:
                    max_acc, max_micro, max_macro = node_classification_evaluate(
                        device_1,
                        enc_feat,
                        args,
                        num_classes,
                        target_type_labels,
                        masked_graph,
                    )
                    if not performance:
                        performance = {"Acc": [], "Micro-F1": [], "Macro-F1": []}
                    performance["Acc"].append(max_acc)
                    performance["Micro-F1"].append(max_micro)
                    performance["Macro-F1"].append(max_macro)
                    log_file.write(
                        "\t [Accuracy:{:4f} Micro-F1:{:4f} Macro-F1:{:4f}  ]\n".format(
                            max_acc,
                            max_micro,
                            max_macro,
                        )
                    )
    print(performance)
    ####
    ## plot the performance
    ####
    if data.has_label_ratio:
        fig, axs = plt.subplots(1, len(data.label_ratio), figsize=(15, 5))
        x_range = list(range(0, args.epoches + 1, args.eva_interval))[1:]
        for i, ratio in enumerate(data.label_ratio):
            axs[i].set_title(f"Performance [Label Rate {ratio}%]")
            axs[i].plot(x_range, performance[ratio]["Acc"], label="Acc")
            axs[i].plot(x_range, performance[ratio]["Macro-F1"], label="Macro-F1")
            axs[i].plot(x_range, performance[ratio]["Micro-F1"], label="Micro-F1")
            axs[i].legend()
            axs[i].set_xlabel("epoch")
        formatted_now = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
        file_name = name_file(args, "img", formatted_now)
        fig.savefig(file_name)
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        x_range = list(range(0, args.epoches + 1, args.eva_interval))[1:]
        axs.set_title(f"Performance")
        axs.plot(x_range, performance["Acc"], label="Acc")
        axs.plot(x_range, performance["Macro-F1"], label="Macro-F1")
        axs.plot(x_range, performance["Micro-F1"], label="Micro-F1")
        axs.legend()
        axs.set_xlabel("epoch")
        formatted_now = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
        file_name = name_file(args, "img", formatted_now)
        fig.savefig(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Heterogeneous Project")
    parser.add_argument("--dataset", type=str, default="dblp")
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eva_interval", type=int, default=10)
    parser.add_argument("--eva_epoches", type=int, default=50)
    parser.add_argument("--num_layer", type=int, default=3, help="number of model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument(
        "--num_out_heads", type=int, default=1, help="number of attention output heads"
    )
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="mask rate")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout probability")
    parser.add_argument("--encoder", type=str, default="HAN", help="heterogeneous encoder")
    parser.add_argument("--decoder", type=str, default="HAN", help="Heterogeneous decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--eva_lr", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--eva_wd", type=float, default=1e-4, help="weight decay for evaluation")
    parser.add_argument("--gamma", type=int, default=3, help="gamma for cosine similarity")
    parser.add_argument("--devices", type=int, default=0, help="gpu devices")
    parser.add_argument("--scheduler", default=True, help="scheduler for optimizer")
    parser.add_argument("--edge_recons", default=True, help="edge reconstruction")
    parser.add_argument("--feat_recons", default=True, help="feature reconstruction")
    parser.add_argument(
        "--all_feat_recons", default=True, help="used all type node feature reconstruction"
    )
    parser.add_argument(
        "--all_edge_recons", default=True, help="use all type node edge reconstruction"
    )
    parser.add_argument("--use_config", default=True, help="use best parameter in config.yaml ")
    known_args, unknow_args = parser.parse_known_args()

    cmd_args = [arg.lstrip("-") for arg in sys.argv[1:] if arg.startswith("--")]
    cmd_args_value = [getattr(known_args, arg) for arg in cmd_args if arg in known_args]
    ## update argument by config file
    if known_args.use_config:
        known_args = load_config(known_args, "config.yaml")

    ## update argument by command line
    for arg, value in zip(cmd_args, cmd_args_value):
        if value == "True" or value == "False":
            value = eval(value)
        setattr(known_args, arg, value)
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    print(known_args)
    train(args=known_args)
