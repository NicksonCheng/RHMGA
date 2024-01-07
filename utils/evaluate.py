import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_classes)

    def forward(self,  x):
        logits = self.linear(x)
        return logits
# MLP Model


class MLP(nn.Module):
    def __init__(self, num_dim, num_classes):
        super(MLP, self).__init__()
        self.hidden = num_dim*2
        self.fc1 = nn.Linear(num_dim, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def score(logits, labels):
    """
    Compute macro,micro f1 score
    """
    _, indices = torch.max(logits, dim=1)
    total = float(indices.size()[0])
    # acc = torch.eq(indices, labels).sum().item() / total
    acc = roc_auc_score(y_true=labels.detach().numpy(),
                        y_score=logits.detach().numpy(), multi_class='ovr')
    return acc, f1_score(labels, indices, average='micro'), f1_score(labels, indices, average='macro')


def mse(x, y):
    return mean_squared_error(x, y)


def cosine_similarity(x, y, gamma):
    # x=tensor(node_num,attribute_dim)

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    cos_sim = (1-(x*y).sum(dim=-1)).pow_(gamma).mean()
    return cos_sim
