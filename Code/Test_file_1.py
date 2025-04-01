import networkx as nx
import matplotlib.pyplot as plt
import random

# Initialize an empty graph
G = nx.Graph()

# Define initial nodes and edges
nodes = [(0, 0), (1, 1), (2, 1), (3, 2)]
edges = [(0, 1), (1, 2)]

# Add initial nodes and edges
for i, pos in enumerate(nodes):
    G.add_node(i, pos=pos)
G.add_edges_from(edges)

# Function to visualize network
def draw_network(G):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.show()

# Expand the network iteratively
for i in range(4, 8):
    new_pos = (random.randint(1, 5), random.randint(1, 5))
    G.add_node(i, pos=new_pos)
    connect_to = random.choice(list(G.nodes))
    G.add_edge(i, connect_to)
    draw_network(G)  # Visualize after each expansion
