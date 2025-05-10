
# In this environment generator functtion, I want to generate multiple simplistic configurations of potential water distribution networks. This will take a 10x10 grid, with a source node in the centre of the network, 3-5 connector nodes around (connected to one another in various configurations) and 3-5 demand nodes at the peripheries.

# import libraries

import os
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import math

# Inputs

source_node = (5, 5)  # Source node at center of a 10x10 grid
num_connect = random.randint(3, 5)  # Number of connector nodes
num_demand = random.randint(3, 5)  # Number of demand nodes
pipe_diameters = [0.1, 0.2, 0.3, 0.4, 0.5]  # Arbitrary pipe diameters in meters
pipe_roughnesses = [0.01, 0.02, 0.03, 0.04, 0.05]  # Arbitrary pipe roughnesses in meters

def generate_network(source_node, num_connect, num_demand, pipe_diameters, pipe_roughnesses):

    """
    Generate a random network of pipes and nodes within a 10x10 grid.
    The source node is at the center of the grid, connector nodes are randomly placed,
    and demand nodes are at the edges of the grid.
    """
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add source node
    G.add_node(source_node, demand=0, source=True)
    
    # Generate connector nodes at random locations in the grid
    connector_nodes = []
    for i in range(num_connect):
        while True:
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            node = (x, y)
            if node != source_node and node not in connector_nodes:
                connector_nodes.append(node)
                G.add_node(node, demand=0, source=False)
                break
    
    # Generate demand nodes at random locations on the edges of the grid
    demand_nodes = []
    edge_locations = [(x, 0) for x in range(10)] + [(x, 9) for x in range(10)] + \
                    [(0, y) for y in range(10)] + [(9, y) for y in range(10)]
    
    for i in range(num_demand):
        while True:
            node = random.choice(edge_locations)
            if node != source_node and node not in connector_nodes and node not in demand_nodes:
                demand_nodes.append(node)
                G.add_node(node, demand=random.choice([1.0, 2.0]), source=False)  # Random demand value
                edge_locations.remove(node)  # Remove this location from available edge locations
                break

    unconnected_nodes = set(connector_nodes)
    # Connect source node to its nearest neighbours
    G, unconnected_nodes, nearest_node = connect_nodes(G, connector_nodes, unconnected_nodes, source_node)

    # Connect each connector node to its nearest neighbouring connector node, in an order than ensures directed flow
    while unconnected_nodes:
        G, unconnected_nodes, nearest_node = connect_nodes(G, connector_nodes, unconnected_nodes, nearest_node)

    # Connect each demand node to its nearest neighbouring connector node
    G = connect_demand_nodes(G, demand_nodes, connector_nodes)

    # Plot the generated network
    plot_network(G)

def connect_nodes(G, node_list, unconnected_nodes, node):
    
    upd_node_list = [n for n in node_list if n != node]  # Exclude the current node from the list
    upd_node_list = [n for n in upd_node_list if not (G.has_edge(node, n) or G.has_edge(n, node))]

    # Connect node to its nearest neighbors in node_list, ensuring the node is not already connected to something and the connection in the opposite direction does not exist
    distances = [(n, np.linalg.norm(np.array(n) - np.array(node))) for n in upd_node_list if n in unconnected_nodes]
    nearest_node = min(distances, key=lambda x: x[1])[0]

    G.add_edge(node, nearest_node, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
    unconnected_nodes.remove(nearest_node)
    return G, unconnected_nodes, nearest_node

def connect_demand_nodes(G, demand_nodes, connector_nodes):
    for demand_node in demand_nodes:
        distances = [(node, np.linalg.norm(np.array(node) - np.array(demand_node))) for node in connector_nodes]
        if distances:
            nearest_connector = min(distances, key=lambda x: x[1])[0]
            G.add_edge(nearest_connector, demand_node, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
    return G

def plot_network(G):
    """
    Plot the generated network using matplotlib with different colors for source, connector, and demand nodes.
    """
    pos = {node: node for node in G.nodes()}
    labels = {node: f"{node}\n{G.nodes[node]['demand']}" for node in G.nodes()}
    
    # Separate nodes by type
    source_nodes = [node for node in G.nodes if G.nodes[node].get('source', False)]
    demand_nodes = [node for node in G.nodes if G.nodes[node].get('demand', 0) > 0]
    connector_nodes = [node for node in G.nodes if node not in source_nodes and node not in demand_nodes]
    
    plt.figure(figsize=(8, 8))
    
    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='red', label='Source Nodes', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=connector_nodes, node_color='blue', label='Connector Nodes', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=demand_nodes, node_color='green', label='Demand Nodes', node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    plt.title("Generated Water Distribution Network")
    plt.legend()
    plt.show()

# Generate and plot the network
generate_network(source_node, num_connect, num_demand, pipe_diameters, pipe_roughnesses)