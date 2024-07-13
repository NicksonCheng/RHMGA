import warnings
import numpy as np
import os
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning


seed = 1
max_iter = 3000
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def cross_validation(edge_embs, edge_labels):

    auc, mrr = [], []
    seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)

    skf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((num_nodes, 1)), np.zeros(num_nodes))):

        print(f"Start Evaluation Fold {fold}!")
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = [], [], [], []
        for each in train_idx:
            train_edge_embs.append(edge_embs[seed_nodes[each]])
            train_edge_labels.append(edge_labels[seed_nodes[each]])
        for each in test_idx:
            test_edge_embs.append(edge_embs[seed_nodes[each]])
            test_edge_labels.append(edge_labels[seed_nodes[each]])
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = (
            np.concatenate(train_edge_embs),
            np.concatenate(test_edge_embs),
            np.concatenate(train_edge_labels),
            np.concatenate(test_edge_labels),
        )

        clf = LinearSVC(random_state=seed, max_iter=max_iter)
        clf.fit(train_edge_embs, train_edge_labels)
        preds = clf.predict(test_edge_embs)
        auc.append(roc_auc_score(test_edge_labels, preds))

        confidence = clf.decision_function(test_edge_embs)
        curr_mrr, conf_num = [], 0
        for each in test_idx:
            test_edge_conf = np.argsort(-confidence[conf_num : conf_num + len(edge_labels[seed_nodes[each]])])
            rank = np.empty_like(test_edge_conf)
            rank[test_edge_conf] = np.arange(len(test_edge_conf))
            curr_mrr.append(1 / (1 + np.min(rank[np.argwhere(edge_labels[seed_nodes[each]] == 1).flatten()])))
            conf_num += len(rank)
        mrr.append(np.mean(curr_mrr))
        assert conf_num == len(confidence)

    return np.mean(auc), np.mean(mrr)


def lp_evaluate(test_file_path, enc_feat):
    enc_feat = enc_feat.cpu()
    dirname = os.path.dirname(test_file_path)
    link_test_file_path = os.path.join(dirname, "link.dat.test")
    target_node_dict = np.load(os.path.join(dirname, "target_node_dict.npy"))
    posi, nega = defaultdict(set), defaultdict(set)

    # map_u = np.where(target_node_dict == 8403)[0].squeeze()
    # map_v1 = np.where(target_node_dict == 14592)[0].squeeze()
    # map_v2 = np.where(target_node_dict == 29699)[0].squeeze()
    # map_v3 = np.where(target_node_dict == 48975)[0].squeeze()
    # map_v4 = np.where(target_node_dict == 23867)[0].squeeze()

    # print(enc_feat[map_u]*(enc_feat[map_v1]))
    # print(enc_feat[map_u]*(enc_feat[map_v2]))
    # print(enc_feat[map_u]*(enc_feat[map_v3]))
    # print(enc_feat[map_u]*(enc_feat[map_v4]))
    # print("-------------------")
    # exit()
    with open(link_test_file_path, "r") as test_file:
        for line in test_file:
            left, right, label = line[:-1].split("\t")
            if label == "1":
                posi[left].add(int(right))
            elif label == "0":
                nega[left].add(int(right))

    edge_embs, edge_labels = defaultdict(list), defaultdict(list)
    for left, rights in posi.items():
        rights = np.array(list(rights))
        map_left = np.where(target_node_dict == int(left))[0].squeeze()
        map_rights = np.where(np.isin(rights, target_node_dict))[0]
        # print(rights.shape,map_rights.shape)
        for map_right in map_rights:
            edge_embs[left].append(enc_feat[map_left]*enc_feat[map_right])
            edge_labels[left].append(1)
    for left, rights in nega.items():
        rights = np.array(list(rights))
        map_left = np.where(target_node_dict == int(left))[0].squeeze()
        map_rights = np.where(np.isin(rights, target_node_dict))[0]
        for map_right in map_rights:
            edge_embs[left].append(enc_feat[map_left]*enc_feat[map_right])
            edge_labels[left].append(0)

    for node in edge_embs:
        edge_embs[node] = np.array(edge_embs[node])
        edge_labels[node] = np.array(edge_labels[node])
        # print(edge_embs[node].shape,edge_labels[node].shape)
    # print(edge_embs)
    # print(edge_labels)
    # pred = []
    # label = []
    # for node in edge_embs:
    #     pred.extend(edge_embs[node])
    #     label.extend(edge_labels[node])

    # # Normalizing scores to range 0 to 1
    # min_score = np.min(pred)
    # max_score = np.max(pred)
    # normalized_scores = [(score - min_score) / (max_score - min_score) for score in pred]
    # print(normalized_scores)
    # auc = roc_auc_score(label, normalized_scores)
    # mrr = average_precision_score(label, normalized_scores)
    auc, mrr = cross_validation(edge_embs, edge_labels)

    return auc, mrr
