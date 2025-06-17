import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Function to ensure a graph is connected
def ensure_connected(G):
    if nx.is_connected(G):
        return G
    
    # Find all connected components
    components = list(nx.connected_components(G))
    
    # Connect all components by adding edges between them
    for i in range(len(components) - 1):
        # Select a random node from current component
        node1 = np.random.choice(list(components[i]))
        # Select a random node from the next component
        node2 = np.random.choice(list(components[i + 1]))
        # Add an edge between these nodes
        G.add_edge(node1, node2)
    
    # Verify the graph is now connected
    assert nx.is_connected(G), "Failed to make graph connected"
    return G

# Generate six different graph models with ≤50 nodes
graphs = {
    "Barabási–Albert (n=20, m=2)": nx.barabasi_albert_graph(20, 2, seed=42),
    "Powerlaw Cluster (n=20, m=2, p=0.1)": nx.powerlaw_cluster_graph(20, 2, 0.1, seed=42),
    "Watts–Strogatz (n=50, k=4, p=0.1)": nx.watts_strogatz_graph(50, 4, 0.1, seed=42),
    "Random Regular (d=3, n=20)": nx.random_regular_graph(3, 20, seed=42),
    "G(n, m) Random (n=20, m=25)": nx.gnm_random_graph(20, 25, seed=42),
    "G(n, p) Random (n=20, p=0.05)": ensure_connected(nx.gnp_random_graph(20, 0.05, seed=42)),
}

# Ensure all graphs are connected
for key in graphs:
    graphs[key] = ensure_connected(graphs[key])

# Create a figure with 2 rows and 3 columns of subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()  # Flatten to make indexing easier

# Draw each graph in its respective subplot
for i, (title, G) in enumerate(graphs.items()):
    ax = axes[i]
    nx.draw(G, ax=ax, node_size=50, with_labels=False)
    # ax.set_title(f"{title}\nConnected: {nx.is_connected(G)}")
    ax.axis('off')  # Turn off axis

plt.tight_layout()  # Adjust spacing between subplots
plt.show()