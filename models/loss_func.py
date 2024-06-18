import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


def mse(x, y):
    return mean_squared_error(x, y)


def cosine_similarity(x, y, gamma):
    # x=tensor(node_num,attribute_dim)

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    cos_sim = (1 - (x * y).sum(dim=-1)).pow_(gamma).mean()
    return cos_sim
