import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import hgt_loader

# Define your heterogeneous graph using HeteroData
data = HeteroData()

# Assuming you have your heterogeneous graph defined already
# Let's create a simple example
# Define node features for two node types
node_features_type1 = torch.randn(3, 16)  # Three nodes of type 1 with 16 features each
node_features_type2 = torch.randn(4, 16)  # Four nodes of type 2 with 16 features each

# Define edge features for two edge types
edge_index_type1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Edge type 1
edge_index_type2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Edge type 2

# Add node features
data["node_features", 0] = node_features_type1
data["node_features", 1] = node_features_type2

# Add edge indices
data["edge_index", (0, 0)] = edge_index_type1  # Edge type 1 connecting nodes of type 0 and 0
data["edge_index", (0, 1)] = edge_index_type2  # Edge type 2 connecting nodes of type 0 and 1

# Create the format compatible with hgt_loader
# You need to extract node features and edge indices for each type
node_features = [data["node_features", i] for i in range(data.num_node_features)]
edge_indices = [
    (data["edge_index", key][0], data["edge_index", key][1]) for key in data.edge_index_keys
]

# Now, you can use hgt_loader
loader = hgt_loader(
    {"node_features": node_features, "edge_indices": edge_indices}, batch_size=1, shuffle=True
)

# Iterate over batches
for batch in loader:
    print(batch)
