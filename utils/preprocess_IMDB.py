import os
import shutil
import zipfile

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset
from dgl.data.utils import (
    download,
    save_graphs,
    save_info,
    load_graphs,
    load_info,
    generate_mask_tensor,
    idx2mask,
)
from dgl.nn.pytorch import MetaPath2Vec
from utils.evaluate import metapath2vec_train


class IMDbDataset(DGLDataset):

    def __init__(self, reverse_edge: bool, use_feat: str, device: int):
        self._ntypes={"m":"movie","a":"actor","d":"director","k":"keywords"}
        self._relations = [
         ("movie", "movie-actor", "actor"),
         ("actor", "actor-movie", "movie"), 
         ("movie", "movie-director", "director"), 
         ("director", "director-movie", "movie"),
         ("movie", "movie-keywords", "keywords"), 
         ("keywords", "keywords-movie", "movie")]
        self.use_feat= use_feat
        self.device=device
        self.data_path = "../data/HeCo/imdb"
        self.g_name=f"imdb_dgl_graph_{self.use_feat}.bin"
        super().__init__("imdb","../data/HeCo")
    def save(self):
        save_graphs(os.path.join(self.data_path, self.g_name), [self.g])
    def load(self):
        print(f"load: {self.data_path}/imdb_dgl_graph{self.use_feat}.bin")
        graphs, _ = load_graphs(os.path.join(self.data_path, self.g_name))
        self.g = graphs[0]
        ntype = self.predict_ntype
        self._num_classes = self.g.nodes[ntype].data["label"].max().item() + 1
        for split in ("train", "val", "test"):
            for ratio in self.label_ratio:
                k = f"{split}_{ratio}"
                self.g.nodes[ntype].data[k] = self.g.nodes[ntype].data[k].bool()

    def process(self):
        print("process")

        self._read_edges()
        origin_feats = self._read_feats()
        if self.use_feat != "origin":
            meta2vec_feat = self._read_meta2vec_feat()
        for ntype, feat in origin_feats.items():
            if self.use_feat == "origin":
                self.g.nodes[ntype].data["feat"] = feat
            elif self.use_feat == "meta2vec":
                self.g.nodes[ntype].data["feat"] = meta2vec_feat[ntype]
            else:
                self.g.nodes[ntype].data["feat"] = torch.hstack([feat, meta2vec_feat[ntype]])

        labels = torch.from_numpy(np.load(os.path.join(self.data_path, "labels.npy"))).long()
        labels=labels-1
        self._num_classes = labels.max().item() + 1
        self.g.nodes[self.predict_ntype].data["label"] = labels
        n = self.g.num_nodes(self.predict_ntype)

        self.masked_graph = {}
        for split in ("train", "val", "test"):
            if split not in self.masked_graph:
                self.masked_graph[split] = {}
            for ratio in self.label_ratio:
                idx = np.load(os.path.join(self.data_path, f"{split}_{ratio}.npy"))
                mask = generate_mask_tensor(idx2mask(idx, n))

                self.g.nodes[self.predict_ntype].data[f"{split}_{ratio}"] = mask

    def _read_edges(self):
        edges = {}
        for file in os.listdir(self.data_path):
            name, ext = os.path.splitext(file)

            if ext == ".txt":
                u, v = name
                e = pd.read_csv(os.path.join(self.data_path, f"{u}{v}.txt"), sep=" ", names=[u, v])
                src = e[u].to_list()
                dst = e[v].to_list()
                src_name, dst_name = self._ntypes[u], self._ntypes[v]
                edges[(src_name, f"{src_name}-{dst_name}", dst_name)] = (src, dst)
                edges[(dst_name, f"{dst_name}-{src_name}", src_name)] = (dst, src)
        self.g = dgl.heterograph(edges)

        for ntype in self._ntypes.values():
            ntype_num_nodes = self.g.num_nodes(ntype)
            ntype_nei_degree = {ntype: torch.zeros(ntype_num_nodes) for ntype in self._ntypes.values()}
            use_relations = [rel_tuple for rel_tuple in self._relations if rel_tuple[2] == ntype]
            for rel_tuple in use_relations:
                src, rel, dst = rel_tuple
                ntype_nei_degree[src] = self.g.in_degrees(torch.arange(ntype_num_nodes), etype=rel)
            self.g.nodes[ntype].data["in_degree"] = torch.stack([ntype_nei_degree[ntype] for ntype in self._ntypes.values()], dim=1)
            # print(self.g.nodes[ntype].data["in_degree"].shape)

    def _read_feats(self):
        feats = {}
        for u in self._ntypes:
            file = os.path.join(self.data_path, f"{u}_feat.npz")
            ntype = self._ntypes[u]
            if os.path.exists(file):
                feats[ntype] = torch.from_numpy(sp.load_npz(file).toarray()).float()
            else:
                num_t = self.g.num_nodes(ntype)
                feats[ntype] = torch.from_numpy(sp.eye(num_t).toarray()).float()
        return feats

    def _read_meta2vec_feat(self):
        ## add metapath2vec feature
        ntype_feats = {}
        for ntype, metapath in self.metapaths.items():
            wd_size = len(metapath) + 1
            metapath_model = MetaPath2Vec(self.g, metapath, wd_size, 512, 3, True)
            metapath2vec_train(self.g, ntype, metapath_model, 50, self.device)

            user_nids = torch.LongTensor(metapath_model.local_to_global_nid[ntype]).to(self.device)
            mp2vec_emb = metapath_model.node_embed(user_nids).detach().cpu()
            ntype_feats[ntype] = mp2vec_emb
            del metapath_model
            torch.cuda.empty_cache()
        return ntype_feats

    def has_cache(self):
        # return False
        return os.path.exists(os.path.join(self.data_path, self.g_name))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only one graph")
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
    @property
    def predict_ntype(self):
        return "movie"
    @property
    def relations(self):
        return self._relations
    @property
    def metapaths(self):
        raise NotImplementedError

    @property
    def evalution_graph(self):
        return self.masked_graph

    @property
    def pos(self):
        return self.pos_i, self.pos_j

    @property
    def has_label_ratio(self):
        return True

    @property
    def multilabel(self):
        return False

    @property
    def label_ratio(self):
        return ["20", "40", "60"]
