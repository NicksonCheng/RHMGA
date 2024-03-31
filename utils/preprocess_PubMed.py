import os
import dgl
import torch
import pandas as pd
from dgl.data import DGLDataset


class PubMedDataset(DGLDataset):
    def __init__(self):
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
        self.graph = dgl.heterograph(self._read_edges())
        self._read_feats()
        label_file = pd.read_csv(
            os.path.join(self.data_path, "label.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_label"],
        )
        print(label_file.shape)
        print(self.graph.nodes[self.predict_ntype].data["feat"])

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
        return edges

    def _read_feats(self):
        feats_file = pd.read_csv(
            os.path.join(self.data_path, "node.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_attributes"],
        )
        for _, row in feats_file.iterrows():
            node_id = int(row["node_id"])
            type_idx = row["node_type"]
            node_type = self._ntypes[type_idx]
            node_feature = [float(x) for x in row["node_attributes"].split(",")]
            if "feat" not in self.graph.nodes[node_type].data:
                num_nodes = self.graph.num_nodes(node_type)
                feat_dim = len(node_feature)
                self.graph.nodes[node_type].data["feat"] = torch.zeros((num_nodes, feat_dim))

            self.graph.nodes[node_type].data["feat"][node_id] = torch.tensor(node_feature)

    def __getitem__(self, i):
        return

    def __len__(self):
        return 1

    @property
    def predict_ntype(self):
        return "Disease"


if __name__ == "__main__":
    data = PubMedDataset()
