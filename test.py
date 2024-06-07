import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F


# Create a simple heterogeneous graph
data = HeteroData()

# Add node features for 'paper' and 'author' node types
data['paper'].x = torch.randn((100, 16))  # 100 papers with 16-dimensional features
data['author'].x = torch.randn((200, 16))  # 200 authors with 16-dimensional features

# Add edges between 'paper' and 'author' nodes
data['paper', 'written_by', 'author'].edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
data['author', 'writes', 'paper'].edge_index = torch.randint(0, 200, (2, 500))  # 500 edges

# Define a NeighborLoader for mini-batch training
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # Sample 10 neighbors at each layer
    batch_size=32,  # Number of target nodes in each batch
    input_nodes=('paper', data['paper'].x)  # Start from 'paper' nodes
)

for batch in loader:
    print(batch.x_dict)
    print(batch.edge_index_dict)
    break