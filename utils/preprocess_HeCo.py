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


class HeCoDataset(DGLDataset):
    """HeCo模型使用的数据集基类

    论文链接：https://arxiv.org/pdf/2105.09111

    类属性
    -----
    * num_classes: 类别数
    * metapaths: 使用的元路径
    * predict_ntype: 目标顶点类型
    * pos: (tensor(E_pos), tensor(E_pos)) 目标顶点正样本对，pos[1][i]是pos[0][i]的正样本

    实现类
    -----
    * ACMHeCoDataset
    * DBLPHeCoDataset
    * FreebaseHeCoDataset
    * AMinerHeCoDataset
    """

    def __init__(self, reverse_edge: bool, use_feat: str, device: int, name, ntypes):
        url = "https://api.github.com/repos/liun-online/HeCo/zipball/main"
        self._ntypes = {ntype[0]: ntype for ntype in ntypes}
        self._label_ratio = ["20", "40", "60"]
        self.use_feat = use_feat
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        super().__init__(name + "-heco", url)

    def download(self):
        file_path = os.path.join(self.raw_dir, "HeCo-main.zip")
        if not os.path.exists(file_path):
            download(self.url, path=file_path)
        with zipfile.ZipFile(file_path, "r") as f:
            f.extractall(self.raw_dir)
        print(self.raw_dir)
        print(self.raw_path)
        shutil.copytree(
            os.path.join(self.raw_dir, "HeCo-main", "data", self.name.split("-")[0]),
            os.path.join(self.raw_path),
        )

    def save(self):
        if self.has_cache():
            return
        save_graphs(os.path.join(self.save_path, self.name + f"_dgl_graph_{self.use_feat}.bin"), [self.g])
        save_info(
            os.path.join(self.raw_path, self.name + "_pos.pkl"),
            {"pos_i": self.pos_i, "pos_j": self.pos_j},
        )

    def load(self):
        print(f"load: {self.save_path}/{self.name}_dgl_graph{self.use_feat}.bin")
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + f"_dgl_graph_{self.use_feat}.bin"))
        self.g = graphs[0]
        ntype = self.predict_ntype
        self._num_classes = self.g.nodes[ntype].data["label"].max().item() + 1
        for split in ("train", "val", "test"):
            for ratio in self.label_ratio:
                k = f"{split}_{ratio}"
                self.g.nodes[ntype].data[k] = self.g.nodes[ntype].data[k].bool()

        info = load_info(os.path.join(self.raw_path, self.name + "_pos.pkl"))
        self.pos_i, self.pos_j = info["pos_i"], info["pos_j"]

    def process(self):
        if self.has_cache():
            return
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

        labels = torch.from_numpy(np.load(os.path.join(self.raw_path, "labels.npy"))).long()
        self._num_classes = labels.max().item() + 1
        self.g.nodes[self.predict_ntype].data["label"] = labels
        n = self.g.num_nodes(self.predict_ntype)

        self.masked_graph = {}
        for split in ("train", "val", "test"):
            if split not in self.masked_graph:
                self.masked_graph[split] = {}
            for ratio in self.label_ratio:
                idx = np.load(os.path.join(self.raw_path, f"{split}_{ratio}.npy"))
                mask = generate_mask_tensor(idx2mask(idx, n))

                self.g.nodes[self.predict_ntype].data[f"{split}_{ratio}"] = mask
        pos_i, pos_j = sp.load_npz(os.path.join(self.raw_path, "pos.npz")).nonzero()
        self.pos_i, self.pos_j = (
            torch.from_numpy(pos_i).long(),
            torch.from_numpy(pos_j).long(),
        )

    def _read_edges(self):
        edges = {}
        for file in os.listdir(self.raw_path):
            name, ext = os.path.splitext(file)

            if ext == ".txt":
                u, v = name
                e = pd.read_csv(os.path.join(self.raw_path, f"{u}{v}.txt"), sep="\t", names=[u, v])
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
            file = os.path.join(self.raw_path, f"{u}_feat.npz")
            ntype=self._ntypes[u]
            if os.path.exists(file):
                feats[ntype] = torch.from_numpy(sp.load_npz(file).toarray()).float()
            else:
                num_t = self.g.num_nodes(ntype)
                feats[ntype]=torch.from_numpy(sp.eye(num_t).toarray()).float()
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
        return False
        return os.path.exists(os.path.join(self.save_path, self.name + f"_dgl_graph_{self.use_feat}.bin"))

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
    def metapaths(self):
        raise NotImplementedError

    @property
    def predict_ntype(self):
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
        return self._label_ratio


class ACMHeCoDataset(HeCoDataset):
    """HeCo模型使用的ACM数据集

    统计数据
    -----
    * 顶点：4019 paper, 7167 author, 60 subject
    * 边：13407 paper-author, 4019 paper-subject
    * 目标顶点类型：paper
    * 类别数：3
    * 顶点划分：60 train, 1000 valid, 1000 test

    paper顶点特征
    -----
    * feat: tensor(N_paper, 1902)
    * label: tensor(N_paper) 0~2
    * train_mask, val_mask, test_mask: tensor(N_paper)

    author顶点特征
    -----
    * feat: tensor(7167, 1902)
    """

    def __init__(self, reverse_edge, use_feat, device):
        
        self._relations = [
            ("paper", "paper-author", "author"),
            ("paper", "paper-subject", "subject"),
            ("author", "author-paper", "paper"),
            ("subject", "subject-paper", "paper"),
        ]
        super().__init__(reverse_edge,use_feat,device,"acm", ["paper", "author", "subject"])
        
    @property
    def metapaths(self):
        return [["pa", "ap"], ["ps", "sp"]]

    @property
    def predict_ntype(self):
        return "paper"

    @property
    def relations(self):
        return self._relations


class DBLPHeCoDataset(HeCoDataset):
    """HeCo模型使用的DBLP数据集

    统计数据
    -----
    * 顶点：4057 author, 14328 paper, 20 conference, 7723 term
    * 边：19645 paper-author, 14328 paper-conference, 85810 paper-term
    * 目标顶点类型：author
    * 类别数：4
    * 顶点划分：80 train, 1000 valid, 1000 test

    author顶点特征
    -----
    * feat: tensor(N_author, 334)
    * label: tensor(N_author) 0~3
    * train_mask, val_mask, test_mask: tensor(N_author)

    paper顶点特征
    -----
    * feat: tensor(14328, 4231)

    term顶点特征
    -----
    * feat: tensor(7723, 50)
    """

    def __init__(self, reverse_edge:bool,use_feat:str, device:int):
        self._relations = [
            ("author", "author-paper", "paper"),
            ("conference", "conference-paper", "paper"),
            ("term", "term-paper", "paper"),
            ("paper", "paper-author", "author"),
            ("paper", "paper-conference", "conference"),
            ("paper", "paper-term", "term"),
        ]
        super().__init__(reverse_edge,use_feat,device, "dblp", ["author", "paper", "conference", "term"])

    def _read_feats(self):
        feats = {}
        for u in "ap":
            file = os.path.join(self.raw_path, f"{u}_feat.npz")
            feats[self._ntypes[u]] = torch.from_numpy(sp.load_npz(file).toarray()).float()
        feats["term"] = torch.from_numpy(np.load(os.path.join(self.raw_path, "t_feat.npz"))).float()
        num_t = self.g.num_nodes("conference")
        feats["conference"]=torch.from_numpy(sp.eye(num_t).toarray()).float()
        return feats

    @property
    def metapaths(self):
        return [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]]

    @property
    def relations(self):
        return self._relations

    @property
    def predict_ntype(self):
        return "author"


class FreebaseHeCoDataset(HeCoDataset):
    """HeCo模型使用的Freebase数据集

    统计数据
    -----
    * 顶点：3492 movie, 33401 author, 2502 director, 4459 writer
    * 边：65341 movie-author, 3762 movie-director, 6414 movie-writer
    * 目标顶点类型：movie
    * 类别数：3 [1327,  618, 1547]
    * 顶点划分：60 train, 1000 valid, 1000 test

    movie顶点特征
    -----
    * label: tensor(N_movie) 0~2
    * train_mask, val_mask, test_mask: tensor(N_movie)
    """

    def __init__(self, reverse_edge: bool, use_feat: str, device: int):
        self._relations = [
            ("author", "author-movie", "movie"),
            ("director", "director-movie", "movie"),
            ("writer", "writer-movie", "movie"),
            ("movie", "movie-author", "author"),
            ("movie", "movie-director", "director"),
            ("movie", "movie-writer", "writer"),
        ]
        super().__init__(reverse_edge, use_feat, device, "freebase", ["movie", "author", "director", "writer"])

    def _read_feats(self):
        feats = {}
        for t in self._ntypes.values():
            num_t = self.g.num_nodes(t)
            feats[t] = torch.from_numpy(sp.eye(num_t).toarray()).float()
        return feats

        # freebased don't have feature, we need to self-define the feature

    @property
    def metapaths(self):
        return {
            "movie": ["movie-author", "author-movie", "movie-director", "director-movie", "movie-writer", "writer-movie"],
            "author": ["author-movie", "movie-author"],
            "director": ["director-movie", "movie-director"],
            "writer": ["writer-movie", "movie-writer"],
        }

    @property
    def relations(self):
        return self._relations

    @property
    def predict_ntype(self):
        return "movie"


class AMinerHeCoDataset(HeCoDataset):
    """HeCo模型使用的AMiner数据集

    统计数据
    -----
    * 顶点：6564 paper, 13329 author, 35890 reference
    * 边：18007 paper-author, 58831 paper-reference
    * 目标顶点类型：paper
    * 类别数：4
    * 顶点划分：80 train, 1000 valid, 1000 test

    movie顶点特征
    -----
    * label: tensor(N_paper) 0~3
    * train_mask, val_mask, test_mask: tensor(N_paper)
    """

    def __init__(self, reverse_edge, use_feat, device):
        self._relations = [
            ("paper", "paper-author", "author"),
            ("paper", "paper-reference", "reference"),
            ("author", "author-paper", "paper"),
            ("reference", "reference-paper", "paper"),
        ]
        super().__init__(reverse_edge,use_feat,device, "aminer", ["paper", "author", "reference"])

    def _read_feats(self):

        feats = {}
        for t in self._ntypes.values():
            num_t = self.g.num_nodes(t)
            feats[t] = torch.from_numpy(sp.eye(num_t).toarray()).float()
        return feats

    @property
    def metapaths(self):
        return [["pa", "ap"], ["pr", "rp"]]

    @property
    def predict_ntype(self):
        return "paper"

    @property
    def relations(self):
        return self._relations
