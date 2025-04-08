
"""
Toy model intended to test the application of Soft Actor Critic (SAC) deep RL in designing optimised water distribution networks. For a simplistic 20x020 grid generate a set of nodes connected by an 'existing network' of branches to reflect an initial state, generate a set of demand nodes in each iteration and connect the existing system. Give the ability to disconnect branches with a penalty. Reward connectivity and punish increased pipe length.
"""

# Packages/Imports
import numpy as np
import networkx as nx
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Independent Variables

iterations = 3
graph_dimensions = (20, 20)
initial_nodes = 6
initial_edges = [initial_nodes - 1, initial_nodes + 4] # Range for potential num nodes

quant_demand_nodes = [3, 5] # How many additional nodes to add with each iteration

# Random
random_seed = 2
random.seed(random_seed)

# Initial Setup

def optimised_wdn(Dimensions, nodes, edges, add_nodes, iterations):
    Initial_graph, initial_nodes, available_positions = generate_initial_wdn(Dimensions, nodes, edges)
    print("Initial available positions: ", len(available_positions))
    Upd_graph, demand_nodes, available_positions = add_demand_nodes(Initial_graph, Dimensions, add_nodes, available_positions)
    upd_nodes = initial_nodes + demand_nodes
    plot_graph(Upd_graph, upd_nodes)

def add_demand_nodes(Graph, Dimensions, add_nodes, available_positions):
    demand_nodes = random.sample(available_positions, random.randint(add_nodes)) # Problem because add nodes is a vector
    available_positions = remove_used_positions(available_positions, demand_nodes)
    Graph.add_nodes_from(demand_nodes)
    return Graph, demand_nodes, available_positions

def generate_initial_wdn(Dimensions, nodes, edges):
    Graph = initialise_grid()
    # Randomly generate positions of nodes and connect to one another
    all_positions = total_positions(Dimensions)
    random_nodes = random.sample(all_positions, nodes)
    Graph.add_nodes_from(random_nodes)
    num_edges = random.randint(edges[0], edges[1])
    possible_edges = list(nx.non_edges(Graph))
    random_edges = random.sample(possible_edges, num_edges)
    Graph.add_edges_from(random_edges)
    available_positions = remove_used_positions(all_positions, random_nodes)
    return Graph, random_nodes, available_positions

# Identify the total list of available positions from the input dimensions
def total_positions(Dimensions):
    rows, cols = Dimensions
    all_positions = [(i, j) for i in range(rows) for j in range(cols)]
    return all_positions

def remove_used_positions(positions, nodes):
    positions = list(set(positions) - set(nodes))
    return positions

def initialise_grid():
    G = nx.Graph()
    return G

# RL model parameters

# Reward function

# Train

# Test

# Graphics
def plot_graph(Graph, nodes):
    plt.figure(figsize=(6, 6))
    pos = {node: (node[1], -node[0]) for node in nodes}
    nx.draw(Graph, pos=pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=6)
    plt.title("Initial_graph")
    plt.grid(True)
    plt.show()

# Run File

optimised_wdn(graph_dimensions, initial_nodes, initial_edges, quant_demand_nodes, iterations)