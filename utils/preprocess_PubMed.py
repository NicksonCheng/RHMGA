import os
import dgl
import torch
import pandas as pd
from dgl.data import DGLDataset
from torch_geometric.data import HeteroData


class PubMedDataset(DGLDataset):
    def __init__(self):
        self.graph = HeteroData()
        self._ntypes = ["Gene", "Disease", "Chemical", "Species"]
        self._relations = [
            ("Gene", "GENE-and-GENE", "Gene"),
            ("Gene", "GENE-causing-DISEASE", "Disease"),
            ("Disease", "DISEASE-and-DISEASE", "Disease"),
            ("Chemical", "CHEMICAL-in-GENE", "Gene"),
            ("Chemical", "CHEMICAL-in-DISEASE", "Disease"),
            ("Chemical", "CHEMICAL-and-CHEMICAL", "Chemical"),
            ("Chemical", "CHEMICAL-in-SPECIES", "Species"),
            ("Species", "SPECIES-with-GENE", "Gene"),
            ("Species", "SPECIES-with-DISEASE", "Disease"),
            ("Species", "SPECIES-and-SPECIES", "Species"),
        ]
        self._classes = [
            "cardiovascular_disease",
            "glandular_disease",
            "nervous_disorder",
            "communicable_disease",
            "inflammatory_disease",
            "pycnosis",
            "skin_disease",
            "cancer",
        ]
        self.data_path = "../data/CKD_data/PubMed"
        super().__init__(name="pubmed")

    def process(self):

        self._read_edges()
        self._read_feats_labels()
        return self.graph

    def _read_edges(self):
        edges = {rel: ([], []) for rel in self._relations}
        edges_file = pd.read_csv(
            os.path.join(self.data_path, "link.dat"),
            sep="\t",
            names=["u_id", "v_id", "link_type", "link_weight"],
        )
        src = edges_file["u_id"].tolist()
        dst = edges_file["v_id"].tolist()
        etype = edges_file["link_type"].tolist()
        for s, d, t in zip(src, dst, etype):
            rel = self._relations[t]
            edges[rel][0].append(s)
            edges[rel][1].append(d)
        for rel, (src, dst) in edges.items():
            src = torch.tensor(src).unsqueeze(0)
            dst = torch.tensor(dst).unsqueeze(0)
            edge_idx = torch.cat((src, dst), dim=0)
            self.graph[rel].edge_index = edge_idx
        return

    def _read_feats_labels(self):
        feats_file = pd.read_csv(
            os.path.join(self.data_path, "node.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_attributes"],
        )
        label_file = pd.read_csv(
            os.path.join(self.data_path, "label.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_label"],
        )
        ## assign node features to graph
        all_node_feat = {}
        pred_node_id = []

        for _, row in feats_file.iterrows():
            node_id = row["node_id"]
            type_idx = row["node_type"]
            node_type = self._ntypes[type_idx]
            node_feature = [float(x) for x in row["node_attributes"].split(",")]
            if node_type not in all_node_feat:
                all_node_feat[node_type] = []

            all_node_feat[node_type].append(node_feature)

            if node_type == self.pred_ntype:
                pred_node_id.append(node_id)

        for ntype, feat in all_node_feat.items():
            self.graph[ntype].x = torch.tensor(feat)

        ## assign pred node label in graph
        pred_num_nodes = self.graph[self.pred_ntype].num_nodes
        self.graph[self.pred_ntype].y = torch.full((pred_num_nodes, 1), -1)

        for _, row in label_file.iterrows():
            node_id = row["node_id"]
            node_type = row["node_type"]
            node_label = row["node_label"]
            if node_id not in pred_node_id:
                # error detection
                continue
            idx = pred_node_id.index(node_id)  ## find index in node feature by node id
            self.graph[self.pred_ntype].y[idx] = torch.tensor([node_label])
        return

    def __getitem__(self, i):
        return

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def pred_ntype(self):
        return "Disease"


if __name__ == "__main__":
    data = PubMedDataset()
