import torch
import numpy as np
import scipy.sparse as sp


def process_data_graph():
    pass


def dblp(label_ratio: list):
    data_path = "../data/HeCo/dblp/"
    # Load labels
    labels = np.load(data_path+'labels.npy', allow_pickle=True)
    labels = torch.LongTensor(labels)

    # load metapath
    metapaths = {
        "apa": sp.load_npz(data_path+"apa.npz"),
        "apcpa": sp.load_npz(data_path+"apcpa.npz"),
        "aptpa": sp.load_npz(data_path+"aptpa.npz")
    }

    # load author's paper neighbor
    nei_p = np.load(data_path+"nei_p.npy", allow_pickle=True)
    # load pos
    pos = sp.load_npz(data_path+"pos.npz")

    # load feature
    feat_a = sp.load_npz(data_path+"a_feat.npz")
    feat_p = sp.load_npz(data_path+"p_feat.npz")
    # feat_t = np.load(data_path+"t_feat.npz")
    # Load train/val/test indices
    train = [np.load(data_path+"train_"+str(i)+".npy") for i in label_ratio]
    val = [np.load(data_path+"val_"+str(i)+".npy") for i in label_ratio]
    test = [np.load(data_path+"test_"+str(i)+".npy") for i in label_ratio]
    return
