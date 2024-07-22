import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils.utils import colorize
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from collections import Counter


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
        return F.softmax(ret, dim=1)


# MLP Model


class MLP(nn.Module):
    def __init__(self, num_dim, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        self.hidden = num_dim * 2
        self.fc1 = nn.Linear(num_dim, self.hidden)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

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


def metapath2vec_train(graph, target_type, model, epoch, device):
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


def node_clustering_evaluate(embeds, y, n_labels, training_time):
    nmi_list, ari_list = [], []
    embeds = embeds.cpu().numpy()
    y = y.cpu().detach().numpy()
    for kmeans_random_state in range(training_time):
        Y_pred = KMeans(n_clusters=n_labels, random_state=kmeans_random_state, n_init=10).fit(embeds).predict(embeds)
        nmi = normalized_mutual_info_score(y, Y_pred)
        ari = adjusted_rand_score(y, Y_pred)
        nmi_list.append(nmi)
        ari_list.append(ari)
    mean = {
        "nmi": np.mean(nmi_list),
        "ari": np.mean(ari_list),
    }
    std = {
        "nmi": np.std(nmi_list),
        "ari": np.std(ari_list),
    }
    return mean, std


def node_classification_evaluate(device, enc_feat, args, num_classes, labels, masked_graph, multilabel, training_time=10):
    train_mask = masked_graph["train"].to(dtype=torch.bool)
    val_mask = masked_graph["val"].to(dtype=torch.bool)
    test_mask = masked_graph["test"].to(dtype=torch.bool)
    emb = {
        "train": enc_feat[train_mask].to(device),
        "val": enc_feat[val_mask].to(device),
        "test": enc_feat[test_mask].to(device),
    }
    labels = {
        "train": labels[train_mask].squeeze().detach().to(device),
        "val": labels[val_mask].squeeze().detach().to(device),
        "test": labels[test_mask].squeeze().detach().to(device),
    }
    accs = []
    micro_f1s = []
    macro_f1s = []
    auc_score_list = []
    for _ in tqdm(range(training_time), position=0, desc=colorize("Evaluating", "green")):

        if args.classifier == "MLP":
            classifier = MLP(num_dim=enc_feat.shape[-1], num_classes=num_classes).to(device)
        elif args.classifier == "LR":
            classifier = LogReg(ft_in=args.num_hidden, nb_classes=num_classes).to(device)
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
        for epoch in range(args.eva_epoches):
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
        "micro_f1": np.mean(micro_f1s),
        "macro_f1": np.mean(macro_f1s),
        "auc_roc": np.mean(auc_score_list),
    }
    std = {
        "micro_f1": np.std(micro_f1s),
        "macro_f1": np.std(macro_f1s),
        "auc_roc": np.std(auc_score_list),
    }
    return mean, std
    classifier.load_state_dict(best_model_state_dict)
    test_output = classifier(multilabel, emb["test"]).cpu()
    test_acc, test_micro_f1, test_macro_f1 = score(test_output, labels["test"], multilabel)
    return test_acc, test_micro_f1, test_macro_f1
    # return max(val_accuracy), max(val_micro), max(val_macro)


def LGS_node_classification_evaluate(device, enc_feat, args, num_classes, labels, masked_graph, multilabel):
    n_split = 10
    labeled_indices = torch.where(masked_graph["total"] > 0)[0]  ## because the mask is a tensor, so we need to use torch.where to get the indices
    ## node all nodes has labels
    labels_dict = labels[labeled_indices].squeeze().detach().cpu()
    enc_feat_dict = enc_feat[labeled_indices].detach().cpu()

    seed = np.random.seed(1)
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    accs = []
    micro_f1s = []
    macro_f1s = []
    auc_roc_list = []  ## baseline model's evaluation
    print(torch.bincount(labels_dict))
    for train_index, test_index in tqdm(skf.split(enc_feat_dict, labels_dict), total=n_split):
        # print(Counter(labels_dict[train_index].tolist()))
        # print("----------------")
        # print(Counter(labels_dict[test_index].tolist()))
        # exit()
        clf = LinearSVC(random_state=seed, max_iter=3000, dual="auto")
        clf.fit(enc_feat_dict[train_index], labels_dict[train_index])
        pred = clf.predict(enc_feat_dict[test_index])
        print(pred)
        print(labels_dict[test_index])
        acc = torch.eq(torch.tensor(pred), labels_dict[test_index]).sum().item() / len(pred)
        pred_score = clf.decision_function(enc_feat_dict[test_index])
        softmax_score = torch.softmax(torch.from_numpy(pred_score), dim=1).numpy()

        macro_f1s.append(f1_score(labels_dict[test_index], pred, average="macro"))
        micro_f1s.append(f1_score(labels_dict[test_index], pred, average="micro"))

        auc_roc_list.append(
            roc_auc_score(
                y_true=labels_dict[test_index],
                y_score=softmax_score,
                multi_class="ovr",
                average=None if multilabel else "macro",
            )
        )
        print(f"Acc:{acc} Macro:{macro_f1s[-1]}, Micro:{micro_f1s[-1]} AUC_ROC:{auc_roc_list[-1]}")
    mean = {
        "auc_roc": np.mean(auc_roc_list),
        "micro_f1": np.mean(micro_f1s),
        "macro_f1": np.mean(macro_f1s),
    }
    std = {
        "auc_roc": np.std(auc_roc_list),
        "micro_f1": np.std(micro_f1s),
        "macro_f1": np.std(macro_f1s),
    }
    return mean, std

    ## my own evaluation
    tqdm_bar = tqdm(total=skf.get_n_splits(enc_feat_dict, labels_dict), position=0, desc=colorize("Evaluating", "green"))
    for i, (train_index, test_index) in enumerate(skf.split(enc_feat_dict, labels_dict)):
        classifier = MLP(num_dim=enc_feat.shape[-1], num_classes=num_classes).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=args.eva_lr, weight_decay=args.eva_wd)
        test_accs = []
        test_micro_f1s = []
        test_macro_f1s = []
        test_out_list = []

        enc_feat_dict = enc_feat_dict.to(device)
        labels_dict = labels_dict.to(device)

        for epoches in range(args.eva_epoches):
            classifier.train()

            train_output = classifier(multilabel, enc_feat_dict[train_index])
            if multilabel:
                eva_loss = F.binary_cross_entropy(train_output, labels_dict[train_index])
            else:
                eva_loss = F.cross_entropy(train_output, labels_dict[train_index])
            optimizer.zero_grad()
            eva_loss.backward()
            optimizer.step()
            with torch.no_grad():
                classifier.eval()
                test_output = classifier(multilabel, enc_feat_dict[test_index])
                test_acc, test_micro_f1, test_macro_f1 = score(test_output, labels_dict[test_index], multilabel)
                test_out_list.append(test_output)
                test_accs.append(test_acc)
                test_micro_f1s.append(test_micro_f1)
                test_macro_f1s.append(test_macro_f1)

        max_inter = test_accs.index(max(test_accs))
        accs.append(test_accs[max_inter])
        max_inter = test_micro_f1s.index(max(test_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_inter])
        max_inter = test_macro_f1s.index(max(test_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_inter])

        best_output = test_out_list[max_inter]
        test_label = labels_dict[test_index].cpu().detach().numpy()
        best_output = best_output.cpu().detach().numpy()
        auc_score = roc_auc_score(
            y_true=test_label,
            y_score=best_output,
            multi_class="ovr",
            average=None if multilabel else "macro",
        )
        auc_roc_list.append(auc_score)

        tqdm_bar.update(1)
    tqdm_bar.close()
    mean = {"acc": np.mean(accs), "micro_f1": np.mean(micro_f1s), "macro_f1": np.mean(macro_f1s), "auc_roc": np.mean(auc_roc_list)}
    std = {"acc": np.std(accs), "micro_f1": np.std(micro_f1s), "macro_f1": np.std(macro_f1s), "auc_roc": np.std(auc_roc_list)}
    return mean, std
