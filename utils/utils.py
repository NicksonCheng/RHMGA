import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def load_config(args, path):
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    config = config[args.dataset]

    for k, v in config.items():
        setattr(args, k, v)
    return args


def colorize(string, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    return f"{colors[color]}{string}{colors['reset']}"


def name_file(args, file, log_times):
    curr_file = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(curr_file, "..")
    if file == "log":
        write_path = os.path.join(file, "performance", args.dataset)
        log_path = os.path.join(project_path, write_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_name = f"./{write_path}/{log_times}_HGARME("
    else:
        write_path = os.path.join(file, args.dataset)
        img_path = os.path.join(project_path, write_path)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        file_name = f"./{write_path}/{log_times}_HGARME("
    if args.edge_recons:
        file_name += "edge"
        if args.all_edge_recons:
            file_name += "[all]"
        file_name += "_"
    if args.feat_recons:
        file_name += "feat"
        if args.all_feat_recons:
            file_name += "[all]"

    file_name += f")_{args.dataset}"
    if file == "log":
        file_name += ".txt"
    else:
        file_name += ".png"
    return file_name

def visualization(embs,labels,log_times,epoch):
    embs=embs.cpu().detach().numpy()
    labels=labels.cpu().detach().numpy()
    perplexity=min(30, embs.shape[0] - 1)
    tsne=TSNE(n_components=2,perplexity=perplexity, random_state=42)
    embs_2d=tsne.fit_transform(embs)
    
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green','yellow','purple','orange','black','pink','brown','gray']
    for label in np.unique(labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(embs_2d[indices, 0],embs_2d[indices, 1], 
                    color=colors[label], label=f'Class {label}', alpha=0.6)
    
    plt.title("t-SNE visualization of node embeddings with class labels")
    plt.xlabel("x t-SNE vector")
    plt.ylabel("y t-SNE vector")
    plt.legend()
    plt.savefig(f"test({epoch})_{log_times}.png")