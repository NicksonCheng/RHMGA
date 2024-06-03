import dgl
import torch

g = dgl.heterograph(
    {
        ("user", "plays", "game"): (torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 1, 1])),
        ("developer", "develops", "game"): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
)
subgs = g[("user", "plays", "game")]


edge_id = subgs.edge_ids(1, 0)
print(edge_id)
subgs.remove_edges(torch.tensor([0]))


print(subgs.edges())


# Example tensor array
tensor_array = torch.tensor([0, 1, 2, 3, 1, 4, 5, 1, 6])
tmp = torch.tensor([2, 4, 6])

# Find indices where values are not equal to 1
indices = tensor_array[tmp]
print(indices)
