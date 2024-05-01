import dgl
import torch

g = dgl.heterograph(
    {
        ("user", "friend", "user"): (torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 1, 1])),
        ("developer", "develops", "game"): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
)
g2 = dgl.remove_edges(g, torch.tensor([0, 3]), "friend")
print(g2.edges(etype="friend"))
print(g.edges(etype="friend"))
