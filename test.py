import torch
import torch.nn.functional as F
import scipy.sparse as sp
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader

# dic1 = {"a": 123, "b": 456, "c": 789}

# dic2 = {key: value + 10 for key, value in dic1.items()}
# print(dic2)


# relations = [("paper", "pa", "author"), ("paper", "pr", "reference"), ("author", "at", "term")]

# t = "author"
# for rel in relations:
#     print(rel if t in rel else None)

# g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2])))

# a1 = torch.zeros(3, 4, 5)
# a2 = torch.zeros(3, 4, 1)
# a3 = a1 * a2
# print(a3.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hg = dgl.heterograph(
    {
        ("user", "like", "movie"): (torch.tensor([0, 1]), torch.tensor([1, 0])),
        ("user", "follow", "user"): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    }
)
train_nids = {
    "user": torch.arange(hg.num_nodes("user")),
    "movie": torch.arange(hg.num_nodes("movie")),
}  # training IDs of 'user' and 'movie' nodes
sampler = MultiLayerFullNeighborSampler(1)  # create a sampler
dataloader = DataLoader(
    hg,
    train_nids,
    sampler,
    batch_size=1,  # batch_size decides how many IDs are passed to sampler at once
    num_workers=4,
)
for i, mini_batch in enumerate(dataloader):
    # unpack the mini batch
    # input_nodes and output_nodes are dictionary while subgs are a list of
    # heterogeneous graphs
    input_nodes, output_nodes, subgs = mini_batch
    print(input_nodes)
    print(output_nodes)
    print(subgs)
