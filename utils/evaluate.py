import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils.utils import colorize
from torch.optim import SparseAdam
from torch.utils.data import DataLoader


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, multilabel, seq):
        ret = self.fc(seq)
        return ret


# MLP Model


class MLP(nn.Module):
    def __init__(self, num_dim, num_classes):
        super(MLP, self).__init__()
        self.hidden = num_dim * 2
        self.fc1 = nn.Linear(num_dim, self.hidden)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden, num_classes)

    def forward(self, multilabel, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        if multilabel:
            return self.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


def score(logits, labels, multilabel):
    logits = logits.cpu()
    labels = labels.cpu()
    """
    Compute macro,micro f1 score
    """
    _, indices = torch.max(logits, dim=1)
    total = float(indices.shape[0])
    acc = torch.eq(indices, labels).sum().item() / total
    # acc = roc_auc_score(y_true=labels.detach().numpy(), y_score=logits.detach().numpy(), multi_class="ovr", average=None if multilabel else "macro")
    return (
        acc,
        f1_score(labels, indices, average="micro"),
        f1_score(labels, indices, average="macro"),
    )


def mse(x, y):
    return mean_squared_error(x, y)


def cosine_similarity(x, y, gamma):
    # x=tensor(node_num,attribute_dim)

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    cos_sim = (1 - (x * y).sum(dim=-1)).pow_(gamma).mean()
    return cos_sim


def metapath2vec_train(args, graph, target_type, model, epoch, device):
    for e in tqdm(range(epoch), desc="Metapath2vec Training"):
        # Use the source node type of etype 'uc'
        dataloader = DataLoader(torch.arange(graph.num_nodes(target_type)), batch_size=128, shuffle=True, collate_fn=model.sample)
        optimizer = SparseAdam(model.parameters(), lr=0.025)
        model.to(device)
        model.train()

        total_loss = 0
        for pos_u, pos_v, neg_v in dataloader:
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)
            loss = model(pos_u, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    return model


def node_clustering_evaluate(embeds, y, n_labels, kmeans_random_state):
    embeds = embeds.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    Y_pred = KMeans(n_clusters=n_labels, random_state=kmeans_random_state, n_init=10).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)

    return nmi, ari


def node_classification_evaluate(device, enc_feat, args, num_classes, labels, masked_graph, multilabel):

    train_mask = masked_graph["train"].to(dtype=torch.bool)
    val_mask = masked_graph["val"].to(dtype=torch.bool)
    test_mask = masked_graph["test"].to(dtype=torch.bool)
    emb = {
        "train": enc_feat[train_mask].to(device),
        "val": enc_feat[val_mask].to(device),
        "test": enc_feat[test_mask].to(device),
    }
    labels = {
        "train": labels[train_mask].squeeze(),
        "val": labels[val_mask].squeeze(),
        "test": labels[test_mask].squeeze(),
    }
    accs = []
    micro_f1s = []
    macro_f1s = []
    auc_score_list = []
    for _ in range(10):

        classifier = MLP(num_dim=enc_feat.shape[-1], num_classes=num_classes).to(device)
        # classifier = LogReg(ft_in=args.num_hidden, nb_classes=num_classes).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=args.eva_lr, weight_decay=args.eva_wd)
        test_outputs_list = []
        val_macro = []
        val_micro = []
        val_accuracy = []
        test_macro = []
        test_micro = []
        test_accuracy = []
        best_val_acc = 0.0
        best_model_state_dict = None
        loss_func = nn.CrossEntropyLoss()
        for epoch in tqdm(range(args.eva_epoches), position=0, desc=colorize("Evaluating", "green")):
            classifier.train()
            train_output = classifier(multilabel, emb["train"])
            if multilabel:
                eva_loss = F.binary_cross_entropy(train_output, labels["train"])
            else:
                eva_loss = loss_func(train_output, labels["train"])
            optimizer.zero_grad()
            eva_loss.backward()
            optimizer.step()

            with torch.no_grad():
                classifier.eval()
                val_output = classifier(multilabel, emb["val"])

                val_acc, val_micro_f1, val_macro_f1 = score(val_output, labels["val"], multilabel)

                test_output = classifier(multilabel, emb["test"])
                test_acc, test_micro_f1, test_macro_f1 = score(test_output, labels["test"], multilabel)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state_dict = classifier.state_dict()

                val_accuracy.append(val_acc)
                val_micro.append(val_micro_f1)
                val_macro.append(val_macro_f1)

                test_macro.append(test_macro_f1)
                test_micro.append(test_micro_f1)
                test_accuracy.append(test_acc)
                test_outputs_list.append(test_output)
        max_iter = val_accuracy.index(max(val_accuracy))
        accs.append(test_accuracy[max_iter])
        max_iter = val_micro.index(max(val_micro))
        micro_f1s.append(test_micro[max_iter])
        max_iter = val_macro.index(max(val_macro))
        macro_f1s.append(test_macro[max_iter])

        best_output = test_outputs_list[max_iter]
        test_label = labels["test"].cpu().detach().numpy()
        best_output = best_output.cpu().detach().numpy()
        auc_score_list.append(
            roc_auc_score(
                y_true=test_label,
                y_score=best_output,
                multi_class="ovr",
                average=None if multilabel else "macro",
            )
        )

    mean = {
        "acc": np.mean(auc_score_list),
        "micro_f1": np.mean(micro_f1s),
        "macro_f1": np.mean(macro_f1s),
        "auc": np.mean(auc_score_list),
    }
    std = {
        "acc": np.std(auc_score_list),
        "micro_f1": np.std(micro_f1s),
        "macro_f1": np.std(macro_f1s),
        "auc": np.std(auc_score_list),
    }
    return mean, std
    classifier.load_state_dict(best_model_state_dict)
    test_output = classifier(multilabel, emb["test"]).cpu()
    test_acc, test_micro_f1, test_macro_f1 = score(test_output, labels["test"], multilabel)
    return test_acc, test_micro_f1, test_macro_f1
    # return max(val_accuracy), max(val_micro), max(val_macro)
