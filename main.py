from datetime import datetime
import time
import os
import sys
import argparse
import signal
import dgl
import numpy as np
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from utils.preprocess_HGB import ACMHGBDataset, FreebaseHGBDataset
from utils.preprocess_HeCo import (
    ACMHeCoDataset,
    AMinerHeCoDataset,
    FreebaseHeCoDataset,
)
from utils.preprocess_IMDB import IMDbDataset
from utils.preprocess_PubMed import PubMedDataset
from utils.evaluate import LGS_node_classification_evaluate, node_classification_evaluate, node_clustering_evaluate, metapath2vec_train
from utils.link_prediction import lp_evaluate
from utils.utils import load_config, colorize, name_file, visualization, save_best_performance
from models.HGARME import HGARME
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

heterogeneous_dataset = {
    "heco_acm": {
        "name": ACMHeCoDataset,
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
    "HGB_Freebase": {"name": FreebaseHGBDataset},
}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def signal_handler(signal, frame):
    print("Received Ctrl+C! Exiting gracefully.")
    dist.destroy_process_group()
    sys.exit(0)


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


def train(rank=None, world_size=None, args=None):

    # device setting and loading dataset
    start_t = time.time()
    if args.parallel:
        setup(rank, world_size)
        signal.signal(signal.SIGINT, signal_handler)  # Register signal handler
    device_0 = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    device_1 = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.devices)
    data = heterogeneous_dataset[args.dataset]["name"](args.reverse_edge, args.use_feat, args.devices)
    print("Preprocessing Time taken:", time.time() - start_t, "seconds")
    start_t = time.time()

    relations = data.relations
    graph = data[0]
    all_types = list(data._ntypes.values())
    target_type = data.predict_ntype
    num_classes = data.num_classes
    train_nids = {ntype: torch.arange(graph.num_nodes(ntype)) for ntype in all_types}
    # for ntype, feat in graph.ndata["feat"].items():
    #     num_nodes = graph.num_nodes(ntype)
    #     graph.nodes[ntype].data["onehot_feat"] = torch.from_numpy(sp.eye(num_nodes).toarray()).float()
    masked_graph = {}
    if data.has_label_ratio:
        for ratio in data.label_ratio:
            masked_graph[ratio] = {}
            for split in ["train", "val", "test"]:
                masked_graph[ratio][split] = graph.nodes[target_type].data[f"{split}_{ratio}"]
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
        num_workers=4,
        use_ddp=args.parallel,
        drop_last=False,
    )
    model = HGARME(
        relations=relations,
        target_type=target_type,
        all_types=all_types,
        target_in_dim=graph.ndata["feat"][target_type].shape[1],
        ntype_in_dim={ntype: feat.shape[1] for ntype, feat in graph.ndata["feat"].items()},
        ntype_out_dim={ntype: feat.shape[1] for ntype, feat in graph.ndata["feat"].items()},
        args=args,
    )
    if args.parallel:
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = model.to(device_0)
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
    log_file_name = name_file(args, "log", log_times)

    ## early stopping strategy
    best_model_dict = None
    best_epoch = 0
    min_loss = 1e8
    wait_count = 0
    # print(f"Model Allocated Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    ## find best performance in model
    if data.has_label_ratio:
        best_nc_performance = {}
        for ratio in data.label_ratio:
            best_nc_performance[ratio] = {
                "auc_roc": 0.0,
                "micro_f1": 0.0,
                "macro_f1": 0.0,
            }
        best_nc_performance["total"] = 0.0
        best_nc_performance["epoch"] = 0
    else:
        best_nc_performance = {"auc_roc": 0.0, "micro_f1": 0.0, "macro_f1": 0.0, "total": 0.0, "epoch": 0}
    best_cls_performance = {"ari": 0.0, "nmi": 0.0, "total": 0.0, "epoch": 0}
    print(f"Num Batches:{len(dataloader)}")
    with dataloader.enable_cpu_affinity():
        for epoch in tqdm(range(args.epoches), total=args.epoches, desc=colorize("Epoch Training", "blue")):
            model.train()
            train_loss = 0.0
            for i, mini_batch in enumerate(dataloader):
                src_nodes, dst_nodes, subgs = mini_batch

                if args.parallel:
                    subgs = [sg.to(rank) for sg in subgs]
                else:
                    subgs = [sg.to(device_0) for sg in subgs]
                curr_mask_rate, feat_loss, adj_loss, degree_loss = model(subgs, relations, epoch, i)
                # print(feat_loss, adj_loss, degree_loss * 0.01)
                loss = args.feat_alpha * feat_loss + args.edge_alpha * adj_loss
                # print(loss)
                train_loss += loss.item()
                loss.backward()
                if not args.accumu_grad:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                #break
            if args.accumu_grad:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(dataloader)

            # Gather losses from all processes
            if args.parallel:
                avg_train_loss = torch.tensor([avg_train_loss], device=rank)
                dist.reduce(avg_train_loss, dst=0, op=dist.ReduceOp.SUM)
                avg_train_loss = avg_train_loss.item() / world_size
            if not args.parallel or rank == 0:
                print(f"Epoch:{epoch+1}/{args.epoches} Training Loss:{(avg_train_loss)} Learning_rate={scheduler.get_last_lr()}")
                total_loss.append(avg_train_loss)
                if avg_train_loss < min_loss:
                    min_loss = avg_train_loss
                    best_model_dict = model.state_dict()
                    best_epoch = epoch + 1
                    wait_count = 0
                else:
                    wait_count += 1

                ## Evaluate Embedding Performance
                if wait_count > args.partience:
                #if epoch > 0 and ((epoch + 1) % args.eva_interval) == 0:
                    model.load_state_dict(best_model_dict)
                    model.eval()
                    log_file_name = name_file(args, "log", log_times)
                    if epoch > 0:
                        with open(log_file_name, "a") as log_file:
                            if os.path.getsize(log_file_name) == 0:
                                log_file.write(f"{args}\n")
                            log_file.write(f"Best Epoches:{best_epoch}-----------------------------------\n")
                            log_file.write("Current Mask Rate:{}\n".format(curr_mask_rate))
                            log_file.write(f"Loss:{min_loss}\n")
                            log_file.close()
                        
                        # enc_feat = features[target_type]
                        if args.parallel:
                            enc_feat, att_sc = model.module.encode_embedding(graph, relations, target_type, "evaluation", rank)
                        else:
                            enc_feat, att_sc = model.encode_embedding(graph, relations, target_type, "evaluation", device_0)
                        target_type_labels = graph.nodes[target_type].data["label"]
                        if args.task == "classification" or args.task == "all":

                            # load model for classification
                            # model.load_state_dict(torch.load(f"analysis/{args.dataset}/best_clustering_[2024-08-07_21:48:57].pth"))
                            # model.eval()
                            # enc_feat, att_sc = model.encode_embedding(
                            #     graph,
                            #     relations,
                            #     target_type,
                            #     "evaluation",
                            # )
                            if data.has_label_ratio:
                                ratio_mean = {}
                                for ratio in data.label_ratio:
                                    for split in masked_graph[ratio].keys():
                                        masked_graph[ratio][split] = masked_graph[ratio][split].detach()
                                    mean, std = node_classification_evaluate(
                                        device_1, enc_feat, args, num_classes, target_type_labels, masked_graph[ratio], data.multilabel
                                    )

                                    if ratio not in performance:
                                        performance[ratio] = {"auc_roc": [], "Micro-F1": [], "Macro-F1": [], "Loss": [], "Epoches": []}
                                    ratio_mean[ratio] = mean
                                    performance[ratio]["auc_roc"].append(mean["auc_roc"])
                                    performance[ratio]["Micro-F1"].append(mean["micro_f1"])
                                    performance[ratio]["Macro-F1"].append(mean["macro_f1"])
                                    performance[ratio]["Loss"].append(min_loss)
                                    performance[ratio]["Epoches"].append(best_epoch)
                                    with open(log_file_name, "a") as log_file:
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
                                        log_file.close()
                                save_best_performance(
                                    model,
                                    best_nc_performance,
                                    ratio_mean,
                                    "classification",
                                    args.dataset,
                                    best_epoch,
                                    log_times,
                                    args.save_model,
                                    True,
                                )

                            else:
                                mean, std = LGS_node_classification_evaluate(
                                    device_1, enc_feat, args, num_classes, target_type_labels, masked_graph, data.multilabel
                                )

                                if not performance:
                                    performance = {"auc_roc": [], "Micro-F1": [], "Macro-F1": [], "Loss": [], "Epoches": []}
                                performance["auc_roc"].append(mean["auc_roc"])
                                performance["Micro-F1"].append(mean["micro_f1"])
                                performance["Macro-F1"].append(mean["macro_f1"])
                                performance["Loss"].append(min_loss)
                                performance["Epoches"].append(best_epoch)
                                with open(log_file_name, "a") as log_file:
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
                                    log_file.close()
                        if args.task == "clustering" or args.task == "all":

                            # load model for clustering
                            # model.load_state_dict(
                            #     torch.load(f"analysis/{args.dataset}/best_clustering_[2024-12-06___12_56_32]", map_location=f"{device_0}")
                            # )
                            # model.eval()
                            # enc_feat, att_sc = model.encode_embedding(graph, relations, target_type, "evaluation", device_0)

                            ## plot t-SNE result
                            emb_2d = visualization(args.dataset, enc_feat, target_type_labels, log_times, best_epoch, args.cls_visual)

                            if not data.has_label_ratio:
                                labeled_indices = torch.where(masked_graph["total"] > 0)[0]
                                enc_feat = enc_feat[labeled_indices]
                                target_type_labels = target_type_labels[labeled_indices].squeeze()
                            cls_enc_emb = enc_feat if args.dataset != "heco_aminer" else emb_2d
                            mean, std = node_clustering_evaluate(args.clus_algorithm,cls_enc_emb, target_type_labels, num_classes, 10)

                            save_best_performance(
                                model, best_cls_performance, mean, "clustering", args.dataset, best_epoch, log_times, args.save_model
                            )
                            with open(log_file_name, "a") as log_file:
                                log_file.write(
                                    "\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]\n".format(
                                        mean["nmi"],
                                        std["nmi"],
                                        mean["ari"],
                                        std["ari"],
                                    )
                                )
                                log_file.close()

                            # else:
                            #     auc, mrr = lp_evaluate(f"data/CKD_data/{args.dataset}/", enc_feat)
                            #     log_file.write(f"\t AUC: {auc} MRR: {mrr}")
                            # elif args.task == "link_prediction":
                            #     auc, mrr = lp_evaluate(data.test_file, enc_feat)
                            #     log_file.write(f"\t AUC: {auc} MRR: {mrr}")
                    ## reset strategy parameter
                    wait_count = 0
                    best_model_dict = None
                    best_epoch = 0
                    min_loss = 1e8
        if args.parallel:
            dist.destroy_process_group()

    ## save the best performance
    with open(log_file_name, "a") as log_file:
        log_file.write("====================================================================\n")
        log_file.write("\t Best Classification Epoches {}: \n".format(best_nc_performance["epoch"]))
        for ratio in data.label_ratio:
            log_file.write(
                "\t Label Rate:{}% Accuracy:[{:.4f}] Micro-F1:[{:.4f}] Macro-F1:[{:.4f}]  \n".format(
                    ratio,
                    best_nc_performance[ratio]["auc_roc"],
                    best_nc_performance[ratio]["micro_f1"],
                    best_nc_performance[ratio]["macro_f1"],
                )
            )
        log_file.write("\n\n\t Best CLustering Epoches {}: \n".format(best_cls_performance["epoch"]))
        log_file.write(
            "\t[clustering] nmi: [{:.4f}] ari: [{:.4f}]\n".format(
                best_cls_performance["nmi"],
                best_cls_performance["ari"],
            )
        )
        log_file.close()
    ####
    ## plot the performance
    ####
    if args.trend_graph:
        if data.has_label_ratio:
            fig, axs = plt.subplots(1, len(data.label_ratio) + 1, figsize=(15, 5))
            x_range = performance[ratio]["Epoches"]
            for i, ratio in enumerate(data.label_ratio):
                axs[i].set_title(f"Performance [Label Rate {ratio}%]")
                axs[i].plot(x_range, performance[ratio]["auc_roc"], label="auc_roc")
                axs[i].plot(x_range, performance[ratio]["Macro-F1"], label="Macro-F1")
                axs[i].plot(x_range, performance[ratio]["Micro-F1"], label="Micro-F1")
                axs[i].legend()
                axs[i].set_xlabel("epoch")
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            x_range = performance[ratio]["Epoches"]
            axs[0].set_title(f"Performance")
            axs[0].plot(x_range, performance["auc_roc"], label="auc_roc")
            axs[0].plot(x_range, performance["Macro-F1"], label="Macro-F1")
            axs[0].plot(x_range, performance["Micro-F1"], label="Micro-F1")
            axs[0].legend()
            axs[0].set_xlabel("epoch")

        x_range = list(range(args.epoches))
        axs[-1].plot(x_range, total_loss, label="Loss")
        axs[-1].legend()
        axs[-1].set_xlabel("epoch")
        formatted_now = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
        img_file_name = name_file(args, "img", formatted_now)
        fig.savefig(img_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Heterogeneous Project")

    ## hyperparameter setting
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
    parser.add_argument("--dynamic_mask_rate", type=str, default="0.4,0.01,0.8", help="dynamic mask rate")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout probability")
    parser.add_argument("--encoder", type=str, default="HAN", help="heterogeneous encoder")
    parser.add_argument("--decoder", type=str, default="HAN", help="Heterogeneous decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--eva_lr", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--eva_wd", type=float, default=1e-4, help="weight decay for evaluation")
    parser.add_argument("--gamma", type=int, default=3, help="gamma for cosine similarity")
    parser.add_argument("--devices", type=int, default=0, help="gpu devices")
    parser.add_argument("--scheduler", default=True, help="scheduler for optimizer")
    parser.add_argument("--partience",type=int,default=5, help = "patients for early stopping")

    ## model setting
    parser.add_argument("--edge_recons", default=True, help="edge reconstruction")
    parser.add_argument("--feat_recons", default=False, help="feature reconstruction")
    parser.add_argument("--degree_recons", default=False, help="degree reconstruction")
    parser.add_argument("--all_feat_recons", default=True, help="used all type node feature reconstruction")
    parser.add_argument("--all_edge_recons", default=True, help="use all type node edge reconstruction")
    parser.add_argument("--all_degree_recons", default=False, help="use all type node degree reconstruction")
    parser.add_argument("--use_config", default=True, help="use best parameter in config.yaml ")
    parser.add_argument("--reverse_edge", default=True, help="add reverse edge or not")
    parser.add_argument("--mp2vec_feat", default=True, help="add reverse edge or not")
    parser.add_argument("--accumu_grad", default=False, help="use accumulate gradient")
    parser.add_argument("--nei_sample", type=str, default="full", help="multilayer neighbor sample")
    parser.add_argument("--use_feat", type=str, default="origin", help="way for original feature construction")
    parser.add_argument("--aggregator", type=str, default="attention", help="way for semantic aggregation")
    parser.add_argument("--feat_alpha", type=float, default=1.0, help="feature reconstruction loss alpha weight")
    parser.add_argument("--edge_alpha", type=float, default=1.0, help="edge reconstruction loss alpha weight")
    parser.add_argument("--edge_mask", default=True, help="mask edge or not")
    parser.add_argument("--feat_mask", default=True, help="mask node feature or not")
    ## event controller
    parser.add_argument("--classifier", type=str, default="MLP", help="classifier for node classification")
    parser.add_argument("--clus_algorithm", type=str, default="K-Means", help="clustering algorithm")
    parser.add_argument("--task", type=str, default="all", help="downstream task")
    parser.add_argument("--cls_visual", default=False, help="draw clustering visualization")
    parser.add_argument("--save_model", default=False, help="save best model or not")
    parser.add_argument("--trend_graph", default=False, help="performance trend graph")
    parser.add_argument("--parallel", default=False, help="whether to user distribute parallel or not")
    parser.add_argument("--save_folder", type=str, default="", help="downstream task")
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

    world_size = torch.cuda.device_count()
    print(known_args)
    if known_args.parallel:
        mp.spawn(train, args=(world_size, known_args), nprocs=world_size, join=True)
    else:
        train(args=known_args)
