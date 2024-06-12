import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: Prepare the graph data
# Create a sample graph
G = nx.karate_club_graph()

# Assume we have class labels for each node
# Here we randomly assign one of three classes to each node as an example
import numpy as np
np.random.seed(42)
labels = np.random.choice([0, 1, 2], size=len(G.nodes()))

# Step 2: Compute node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings for each node
embeddings = []
nodes = []
for node in G.nodes():
    embeddings.append(model.wv[str(node)])
    nodes.append(node)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Step 3: Apply t-SNE to reduce dimensionality to 2D
perplexity = min(30, len(nodes) - 1)  # Ensure perplexity is less than number of nodes
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Step 4: Visualize the results
plt.figure(figsize=(12, 8))

# Define colors for the 3 classes
colors = ['red', 'green', 'blue']

# Plot each class with a different color
for label in np.unique(labels):
    indices = [i for i, lbl in enumerate(labels) if lbl == label]
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                color=colors[label], label=f'Class {label}', alpha=0.6)

# Optionally annotate nodes
for i, node in enumerate(nodes):
    plt.annotate(str(node), (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("t-SNE visualization of Karate Club node embeddings with class labels")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.legend()
plt.savefig("test.png")
