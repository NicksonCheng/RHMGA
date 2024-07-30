import os
import dgl
import torch
import pandas as pd
import numpy as np
from dgl.data import DGLDataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor, idx2mask
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
class HGBDataset(DGLDataset):
    def __init__(
        self, reverse_edge,name,ntypes, use_feat: str = "meta2vec", device: int = 0, url:bool=None, raw_dir:bool=None, save_dir:bool=None, force_reload:bool=False, verbose:bool=False
    ):
        self.graph=HeteroData()
        self.curr_dir = os.path.dirname(__file__)
        self.parent_dir = os.path.dirname(self.curr_dir)
        self.data_path = os.path.join(self.parent_dir, f"data/HGB/{name}")
        super(HGBDataset, self).__init__(
            name="hgb",
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )
    def download(self):
        pass
    def load(self):
        pass
    def save(self):
        pass
    def process(self):
        pass
class ACMHGBDataset(HGBDataset):
    def __init__(self,reverse_edge,use_feat,device):
        self.relations=[
            ("Paper","Paper-cite-Paper","Paper")
            ("Paper","Paper-ref-Paper","Paper")
            ("Paper","Paper-Author","Author")
            ("Paper","Paper-Subject","Subject")
            ("Paper","Paper-Term","Term")
        ]
        super().init__(reverse_edge,"ACM",["Paper","Author","Subject","Term"],use_feat,device)


class FreebaseHGBDataset(HGBDataset):
    def __init__(self,reverse_edge,use_feat,device):
        super().init__(reverse_edge,"Freebase",["Book", "Film", "Music", "Sports", "People", "Location", "Organization","Business"],use_feat,device)