import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def score(logits, labels):
    """
    Compute macro,micro f1 score
    """
    _, indices = torch.max(logits, dim=1)
    return f1_score(labels, indices, average='micro'), f1_score(labels, indices, average='macro')


def cosine_similarity(x, y, gamma):
    # x=tensor(node_num,attribute_dim)

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    cos_sim = (1-(x*y).sum(dim=-1)).pow_(gamma).mean()
    return cos_sim
