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
from dgl.nn.pytorch import MetaPath2Vec
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from utils.preprocess_DBLP import DBLP4057Dataset, DBLPFourAreaDataset
from utils.preprocess_ACM import ACMDataset
from utils.preprocess_HeCo import (
    DBLPHeCoDataset,
    ACMHeCoDataset,
    AMinerHeCoDataset,
    FreebaseHeCoDataset,
)
from utils.preprocess_IMDB import IMDbDataset
from utils.preprocess_Freebase import FreebaseDataset
from utils.preprocess_Yelp import YelpDataset
from utils.preprocess_PubMed import PubMedDataset
from utils.evaluate import LGS_node_classification_evaluate, node_classification_evaluate, node_clustering_evaluate, metapath2vec_train

from utils.utils import load_config, colorize, name_file, visualization
from models.HGARME import HGARME
from models.HAN import HAN
from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE

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
    "heco_dblp": {
        "name": DBLPHeCoDataset,
    },
    "heco_freebase": {
        "name": FreebaseHeCoDataset,
    },
    "heco_aminer": {
        "name": AMinerHeCoDataset,
    },
    "imdb": {
        "name": IMDbDataset,
    },
    "PubMed": {
        "name": PubMedDataset,
    },
    "Yelp": {
        "name": YelpDataset,
    },
    "Freebase": {
        "name": FreebaseDataset,
    },
}


def node_importance(graph):
    split_idx = [feat.shape[0] for feat in graph.ndata["feat"].values()]
    nx_g = dgl.to_networkx(graph, node_attrs=["feat"])
    ntype_node = nx_g.nodes(data=True)
    num_nodes = graph.num_nodes()

    degree_centrality = nx.degree_centrality(nx_g)
    betweenness_centrality = nx.betweenness_centrality(nx_g)
    closeness_centrality = nx.closeness_centrality(nx_g)
    # eigen_centrality = nx.eigenvector_centrality(nx_g)
    pagerank = nx.pagerank(nx_g)

    nodes_score = {
        "degree_centrality": torch.tensor([degree_centrality.get(i, None) for i in range(num_nodes)]),
        "betweenness_centrality": torch.tensor([betweenness_centrality.get(i, None) for i in range(num_nodes)]),
        "closeness_centrality": torch.tensor([closeness_centrality.get(i, None) for i in range(num_nodes)]),
        # "eigen_centrality": torch.tensor([eigen_centrality.get(i, None) for i in range(num_nodes)]),
        "pagerank": torch.tensor([pagerank.get(i, None) for i in range(num_nodes)]),
    }
    nodes_score["importance"] = sum(nodes_score.values())
    ntype_score = {
        "degree_centrality": {},
        "betweenness_centrality": {},
        "closeness_centrality": {},
        # "eigen_centrality": {},
        "pagerank": {},
        "importance": {},
    }

    curr_idx = 0
    for i, ntype in enumerate(graph.ndata["feat"].keys()):
        for score_name in ntype_score.keys():
            ntype_score[score_name][ntype] = nodes_score[score_name][curr_idx : curr_idx + split_idx[i]]
        curr_idx += split_idx[i]
    torch.save(ntype_score, "freebase_score.pth")
    return 0


def train(args):
    torch.cuda.set_per_process_memory_fraction(1.0, device=f"cuda:{args.devices}")  # limit memory usage
    start_t = time.time()
    # torch.cuda.set_device()
    device_0 = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    device_1 = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    data = heterogeneous_dataset[args.dataset]["name"](args.reverse_edge)
    print("Preprocessing Time taken:", time.time() - start_t, "seconds")
    start_t = time.time()
    relations = data.relations
    # node_imp = node_importance(data[0])
    # exit()

    graph = data[0].to(device_0)
    all_types = list(data._ntypes.values())
    target_type = data.predict_ntype

    if args.mp2vec_feat:
        ## add metapath2vec feature
        metapath = ["movie-author", "author-movie", "movie-director", "director-movie", "movie-writer", "writer-movie"]

        metapath_model = MetaPath2Vec(graph, metapath, 4, args.num_hidden, 3, True)
        metapath2vec_train(args, graph, target_type, metapath_model, 50, device_0)

        user_nids = torch.LongTensor(metapath_model.local_to_global_nid[target_type]).to(device_0)
        mp2vec_emb = metapath_model.node_embed(user_nids).detach()

        original_feat = graph.nodes[target_type].data["feat"]
        graph.nodes[target_type].data["feat"] = mp2vec_emb

        graph.nodes[target_type].data["feat"] = torch.hstack([original_feat, mp2vec_emb])
        ## free memory
        del metapath_model
        torch.cuda.empty_cache()

    num_classes = data.num_classes
    features = {t: graph.nodes[t].data["feat"] for t in all_types if "feat" in graph.nodes[t].data}
    train_nids = {ntype: torch.arange(graph.num_nodes(ntype)).to(device_0) for ntype in all_types}

    masked_graph = {}
    if data.has_label_ratio:
        for ratio in data.label_ratio:
            masked_graph[ratio] = {}
            for split in ["train", "val", "test"]:
                masked_graph[ratio][split] = graph.nodes[target_type].data[f"{split}_{ratio}"]
    elif args.dataset == "imdb":

        for split in ["train", "val", "test"]:
            masked_graph[split] = graph.nodes[target_type].data[split]
    else:
        masked_graph["total"] = graph.nodes[target_type].data["total"]
    # print(features)
    if args.nei_sample == "full":
        sampler = MultiLayerFullNeighborSampler(args.num_layer + 1)
    else:
        nei_sample = [int(nei) for nei in args.nei_sample.split(",")]
        sampler = MultiLayerNeighborSampler(nei_sample)
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
    # Define learnable weights for the loss components
    raw_weights = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = optim.Adam(list(model.parameters()) + [raw_weights], lr=args.lr, weight_decay=args.weight_decay)

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
    log_times = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
    file_name = name_file(args, "log", log_times)

    best_model_dict = None
    best_epoches = 0
    min_loss = 1e8
    wait_count = 0
    # print(f"Model Allocated Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    with open(file_name, "a") as log_file:
        log_file.write(f"{str(args)}\n")
    for epoch in tqdm(range(args.epoches), total=args.epoches, desc=colorize("Epoch Training", "blue")):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, mini_batch in enumerate(dataloader):

            src_nodes, dst_nodes, subgs = mini_batch

            # print("----------------------------------")
            # for rels_tuple in relations:
            #     print(f"{rels_tuple}: {subgs[1].num_edges(rels_tuple)}")
            # print("----------------------------------")
            feat_loss, adj_loss = model(subgs, relations, epoch)
            # weight = torch.softmax(raw_weights, dim=0)
            # edge_weight, feat_weight = weight[0], weight[1]
            feat_weight = 1
            edge_weight = 1
            loss = feat_weight * feat_loss + edge_weight * adj_loss
            train_loss += loss.item()

            loss.backward()

            # print(f"Batch {i} Loss: {loss.item()}")
            # print(f"Batch {i} Allocated Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            # break
        optimizer.step()
        scheduler.step()
        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch:{epoch+1}/{args.epoches} Training Loss:{(avg_train_loss)} Learning_rate={scheduler.get_last_lr()}")
        total_loss.append(avg_train_loss)
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            best_model_dict = model.state_dict()
            best_epoches = epoch + 1
            torch.save(model.state_dict(), f"best_{args.dataset}.pth")
        ## Evaluate Embedding Performance
        if epoch > 0 and ((epoch + 1) % args.eva_interval) == 0:
            model.eval()
            # print(f"feat weight{feat_weight.item()} edge weight{edge_weight.item()}")
            # if True:
            file_name = name_file(args, "log", log_times)
            with open(file_name, "a") as log_file:
                log_file.write(f"Epoches:{epoch}-----------------------------------\n")
                log_file.write(f"Loss:{avg_train_loss}\n")
                # enc_feat = features[target_type]
                enc_feat, att_mp = model.encode_embedding(
                    graph,
                    relations,
                    target_type,
                    features,
                    "evaluation",
                )
                target_type_labels = graph.nodes[target_type].data["label"]
                if args.task == "classification":
                    if data.has_label_ratio:
                        for ratio in data.label_ratio:
                            for split in masked_graph[ratio].keys():
                                masked_graph[ratio][split] = masked_graph[ratio][split].detach()
                            mean, std = node_classification_evaluate(
                                device_1, enc_feat, args, num_classes, target_type_labels.detach(), masked_graph[ratio], data.multilabel
                            )

                            if ratio not in performance:
                                performance[ratio] = {"Acc": [], "Micro-F1": [], "Macro-F1": []}
                            performance[ratio]["Acc"].append(mean["acc"])
                            performance[ratio]["Micro-F1"].append(mean["micro_f1"])
                            performance[ratio]["Macro-F1"].append(mean["macro_f1"])
                            log_file.write(
                                "\t Label Rate:{}% Accuracy:[{:.4f}, {:.4f}] Micro-F1:[{:.4f}, {:.4f}] Macro-F1:[{:.4f}, {:.4f}]  \n".format(
                                    ratio,
                                    mean["auc_roc"],
                                    std["auc_roc"],
                                    mean["micro_f1"],
                                    std["micro_f1"],
                                    mean["macro_f1"],
                                    std["macro_f1"],
                                )
                            )

                    else:
                        if args.dataset == "imdb":
                            mean, std = node_classification_evaluate(
                                device_1, enc_feat, args, num_classes, target_type_labels, masked_graph, data.multilabel
                            )
                        else:
                            mean, std = LGS_node_classification_evaluate(
                                device_1, enc_feat, args, num_classes, target_type_labels, masked_graph, data.multilabel
                            )

                        if not performance:
                            performance = {"Acc": [], "Micro-F1": [], "Macro-F1": []}
                        performance["Acc"].append(mean["auc_roc"])
                        performance["Micro-F1"].append(mean["micro_f1"])
                        performance["Macro-F1"].append(mean["macro_f1"])
                        log_file.write(
                            "\t ACC:[{:4f},{:4f}] Micro-F1:[{:.4f}, {:.4f}] Macro-F1:[{:.4f}, {:.4f}]  \n".format(
                                mean["auc_roc"],
                                std["auc_roc"],
                                mean["micro_f1"],
                                std["micro_f1"],
                                mean["macro_f1"],
                                std["macro_f1"],
                            )
                        )
                        # log_file.write(
                        #     "\t Accuracy:[{:.4f}, {:.4f}] Micro-F1:[{:.4f}, {:.4f}] Macro-F1:[{:.4f}, {:.4f}]  \n".format(
                        #         mean["auc_roc"],
                        #         std["auc_roc"],
                        #         mean["micro_f1"],
                        #         std["micro_f1"],
                        #         mean["macro_f1"],
                        #         std["macro_f1"],
                        #     )
                        # )
                elif args.task == "clustering":
                    nmi_list, ari_list = [], []

                    if not data.has_label_ratio:
                        labeled_indices = torch.where(masked_graph["total"] > 0)[0]
                        enc_feat = enc_feat[labeled_indices]
                        target_type_labels = target_type_labels[labeled_indices].squeeze()

                    for kmeans_random_state in range(10):
                        nmi, ari = node_clustering_evaluate(enc_feat, target_type_labels, num_classes, kmeans_random_state)
                        nmi_list.append(nmi)
                        ari_list.append(ari)
                    log_file.write(
                        "\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(
                            np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)
                        )
                    )

                    ## plot t-SNE result
                    visualization(enc_feat, target_type_labels, log_times, epoch)

    ####
    ## plot the performance
    ####
    if data.has_label_ratio:
        fig, axs = plt.subplots(1, len(data.label_ratio) + 1, figsize=(15, 5))
        x_range = list(range(0, args.epoches + 1, args.eva_interval))[1:]
        for i, ratio in enumerate(data.label_ratio):
            axs[i].set_title(f"Performance [Label Rate {ratio}%]")
            axs[i].plot(x_range, performance[ratio]["Acc"], label="Acc")
            axs[i].plot(x_range, performance[ratio]["Macro-F1"], label="Macro-F1")
            axs[i].plot(x_range, performance[ratio]["Micro-F1"], label="Micro-F1")
            axs[i].legend()
            axs[i].set_xlabel("epoch")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        x_range = list(range(0, args.epoches + 1, args.eva_interval))[1:]
        axs[0].set_title(f"Performance")
        axs[0].plot(x_range, performance["Acc"], label="Acc")
        axs[0].plot(x_range, performance["Macro-F1"], label="Macro-F1")
        axs[0].plot(x_range, performance["Micro-F1"], label="Micro-F1")
        axs[0].legend()
        axs[0].set_xlabel("epoch")

    x_range = list(range(args.epoches))
    axs[-1].plot(x_range, total_loss, label="Loss")
    axs[-1].legend()
    axs[-1].set_xlabel("epoch")
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
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of attention output heads")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--mask_rate", type=float, default=0.4, help="mask rate")
    parser.add_argument("--dynamic_mask_rate", type=str, default="0.4,0.003,0.8", help="dynamic mask rate")
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
    parser.add_argument("--all_feat_recons", default=True, help="used all type node feature reconstruction")
    parser.add_argument("--all_edge_recons", default=True, help="use all type node edge reconstruction")
    parser.add_argument("--use_config", default=True, help="use best parameter in config.yaml ")
    parser.add_argument("--reverse_edge", default=True, help="add reverse edge or not")
    parser.add_argument("--mp2vec_feat", default=True, help="add reverse edge or not")
    parser.add_argument("--task", default="classification", help="downstream task")
    parser.add_argument("--nei_sample", type=str, default="full", help="multilayer neighbor sample")
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
