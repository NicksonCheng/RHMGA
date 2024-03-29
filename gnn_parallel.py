import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import CoraGraphDataset
from torch.utils.data import DataLoader
import torch.nn.parallel as parallel


# Define the GAT model
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": torch.leaky_relu(a)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = torch.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h, g):
        z = self.fc(h)
        g.ndata["z"] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, out_dim)

    def forward(self, h, g):
        h = torch.relu(self.layer1(h, g))
        h = self.layer2(h, g)
        return h


# Load Cora dataset
dataset = CoraGraphDataset()
g = dataset[0]

# Preprocess graph data
features = g.ndata["feat"]
labels = g.ndata["label"]
train_mask = g.ndata["train_mask"]

# Split data across available GPUs
device_ids = [0, 1]  # Modify this according to your system
per_device = len(features) // len(device_ids)
split_features = [features[i * per_device : (i + 1) * per_device] for i in range(len(device_ids))]
split_labels = [labels[i * per_device : (i + 1) * per_device] for i in range(len(device_ids))]
split_train_mask = [
    train_mask[i * per_device : (i + 1) * per_device] for i in range(len(device_ids))
]

# Define data loaders for each GPU
train_loaders = []
for i, (feat, lbl, mask) in enumerate(zip(split_features, split_labels, split_train_mask)):
    train_loader = DataLoader(list(zip(feat, lbl, mask)), batch_size=1, shuffle=True)
    train_loaders.append(train_loader)

# Define the GAT model for each GPU
gat_models = [
    GAT(in_dim=features.shape[1], hidden_dim=64, out_dim=dataset.num_classes) for _ in device_ids
]

# Move models to respective GPUs
for i, model in enumerate(gat_models):
    model.to(device_ids[i])

# Define loss function and optimizer for each GPU
criterions = [nn.CrossEntropyLoss() for _ in device_ids]
optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in gat_models]


# Training loop
def train():
    for epoch in range(num_epochs):
        for i, (train_loader, model, criterion, optimizer) in enumerate(
            zip(train_loaders, gat_models, criterions, optimizers)
        ):
            model.train()
            for batch, data in enumerate(train_loader):
                features, labels, mask = data
                features, labels, mask = (
                    features.to(device_ids[i]),
                    labels.to(device_ids[i]),
                    mask.to(device_ids[i]),
                )
                output = model(features, g)
                loss = criterion(output[mask], labels[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, GPU {i}, Batch {batch}, Loss: {loss.item()}")


# Define number of epochs
num_epochs = 10

# Train the models
train()
