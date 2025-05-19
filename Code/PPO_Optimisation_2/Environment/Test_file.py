
"""

In this file, we test certain functions before implementing them in the main code.

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import math
import random

# Generate 3 test graph
graph1 = nx.random_graphs.erdos_renyi_graph(10, 0.5) # num nodes, prob of edge creation
graph2 = nx.random_graphs.barabasi_albert_graph(10, 2) # num nodes, num edges to attach from a new node to existing nodes
graph3 = nx.random_graphs.watts_strogatz_graph(10, 2, 0.5) # num nodes, num neighbours, prob of rewiring
graph4 = nx.random_graphs.random_lobster(5, 0.2, 0.2) # num nodes, proba of adding edge to backbone, prob of adding edge one level beyond
graph5 = nx.random_graphs.powerlaw_cluster_graph(10, 2, 0.5) # num nodes, num edges to attach from a new node to existing nodes, prob of rewiring

# Visualise the 3 graphs in subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 5))
axs[0, 0].set_title('Erdos-Renyi Graph')
nx.draw(graph1, ax=axs[0, 0], with_labels=True)
axs[0, 1].set_title('Barabasi-Albert Graph')
nx.draw(graph2, ax=axs[0, 1], with_labels=True)
axs[1, 0].set_title('Watts-Strogatz Graph')
nx.draw(graph3, ax=axs[1, 0], with_labels=True)
axs[1, 1].set_title('Random Lobster Graph')
nx.draw(graph4, ax=axs[1, 1], with_labels=True)
axs[1, 2].set_title('Powerlaw Cluster Graph')
nx.draw(graph5, ax=axs[1, 2], with_labels=True)
plt.tight_layout()
plt.show()

"""

From the tests above, the random lobster graphs and powerlaw clustrer graphs seem like good candidates to describe branched and looped networks respectively. The random lobster graphs create a backbone (that could be considered the main pipe) and then add edges to the backbone and one level beyond. The powerlaw cluster graph creates a scale-free network with a high clustering coefficient, which is characteristic of many real-world networks, including water distribution systems.

"""