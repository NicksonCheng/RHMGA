import dgl
import torch
from dgl.dataloading import DataLoader, NeighborSampler

# Define your heterogeneous graph
# Example graph creation (replace with your own data)
g = dgl.heterograph(
    {
        ("user", "like", "movie"): ([0, 1, 2], [0, 1, 2]),
        ("user", "follow", "user"): ([0, 1, 2], [1, 2, 0]),
        ("movie", "liked-by", "user"): ([0, 1, 2], [0, 1, 2]),
    }
)

# Define training node IDs for 'user' and 'movie' nodes
# Example: Train on all nodes
train_nids = {"user": torch.arange(g.num_nodes("user")), "movie": torch.arange(g.num_nodes("movie"))}

# Define sampler
sampler = NeighborSampler([10, 15])

# Create DataLoader
dataloader = DataLoader(g, train_nids, sampler, batch_size=32, shuffle=True, drop_last=False, num_workers=0, use_uva=False)  # Shuffle if needed

# Iterate over batches in the DataLoader
for i, mini_batch in enumerate(dataloader):
    input_nodes, output_nodes, subgs = mini_batch
    print(input_nodes)
    print(output_nodes)
    # Process your batch here
