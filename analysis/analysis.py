import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import dgl
import numpy as np
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
import matplotlib.pyplot as plt

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
    "Yelp": {
        "name": YelpDataset,
    },
    "Freebase": {
        "name": FreebaseDataset,
    },
}

dataset = "heco_acm"
data = heterogeneous_dataset[dataset]["name"](reverse_edge=True, use_feat="origin", device=0)
graph = data[0]
relations = data.relations
target_type = data.predict_ntype
num_classes = data.num_classes
target_relations = [rel_tuple for rel_tuple in relations if rel_tuple[0] == target_type]

labels = graph.nodes[target_type].data["label"]

labels = labels[labels != -1]

sorted_label_indices = torch.argsort(labels)  # sorted label node indices
sorted_labels = labels[sorted_label_indices]  # sorted labels
pivots = torch.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
pivots = torch.cat([torch.tensor([0]), pivots, torch.tensor([len(labels)])])
color = []
color = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "cyan"]
num_row = round(num_classes / 4)
num_col = round(num_classes / num_row)
print(num_row, num_col)
fig, axs = plt.subplots(num_row, num_col, figsize=(30, 15))
print(pivots)
print(target_relations)
# Ensure axs is always a 2D array
if num_row == 1:
    axs = axs[np.newaxis, :]  # Convert to 2D array if only one row
if num_col == 1:
    axs = axs[:, np.newaxis]  # Convert to 2D array if only one column
for i, p in enumerate(pivots):
    if i == pivots.shape[0] - 1:
        break
    h, t = pivots[i], pivots[i + 1]
    label_indices = sorted_label_indices[h:t]
    x_size = label_indices.shape[0]
    nei_type = []
    stack_list = []

    row = i // 4
    col = i % 4
    for j, rel_tuple in enumerate(target_relations):
        v, rel, u = rel_tuple
        nei_type.append(u)
        ntype_num_nei_list = []
        for index in label_indices:
            nei_indices = graph.successors(index, etype=rel)
            ntype_num_nei_list.append(nei_indices.shape[0])
        stack_list.append(ntype_num_nei_list)
        axs[row, col].plot(range(x_size), ntype_num_nei_list)
        # print(rel_tuple, len(ntype_num_nei_list))
    x = label_indices.tolist()

    axs[row, col].set_title(f"neighbor {v} degree in class {i}")
    axs[row, col].legend(nei_type)
fig.savefig(f"heco_acm/{dataset}_neighbor_degree.png")
