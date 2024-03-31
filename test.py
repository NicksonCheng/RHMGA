import dgl
import torch

g = dgl.heterograph(
    {
        ("user", "follows", "user"): ([0, 1], [0, 1]),
        ("developer", "develops", "game"): ([0, 1], [0, 2]),
    }
)

# print(g.adj(etype="develops"))
