import dgl
import torch


import torch

# Example tensor
tensor = torch.tensor([1, 2, 2, 5, 3, 3, 4, 4, 4, 4])

# Count occurrences of each element
counts = torch.bincount(tensor)

print(counts)
