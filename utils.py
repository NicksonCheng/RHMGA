import torch
from sklearn.metrics import f1_score


def f1_score(logits, labels):
    """
    Compute macro,micro f1 score
    """
    _, indices = torch.max(logits, dim=1)
    return f1_score(labels, indices, average='micro'), f1_score(labels, indices, average='macro')
