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


def cross_entropy_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    # neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss
    return pos_loss + neg_loss
