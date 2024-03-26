import torch
import torch.nn.functional as F
import scipy.sparse as sp
import dgl

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

t = torch.tensor([1, 2], device=device)
