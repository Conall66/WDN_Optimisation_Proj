
# In this file, we generate networks of pipes and nodes using the networkx package to feed into main file.

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Input Parameters
num_connect = 5 # Number of connector nodes
num_demand = 5 # Number of demand nodes
pipe_diameters = [0.1, 0.2, 0.3, 0.4, 0.5] # Pipe diameters in meters
pipe_roughnesses = [0.01, 0.02, 0.03, 0.04, 0.05] # Pipe roughnesses in meters
redundant_pipes = 2 # Number of redundant pipes
demands = [10, 20, 30, 40, 50] # Demand values in liters per second

def generate_network(num_connect, num_demand, pipe_diameters, pipe_roughnesses, redundant_pipes, demands):

    # Within a 20 x 20 grid, generate a random network of pipes and nodes. There must be one source node, connector nodes and demand nodes (at the edges of the grid). The source node is at (0,0). The connector nodes are at random locations in the grid. The pipes connect the source node to the connector nodes and the connector nodes to the demand nodes.

    source_node = (10, 10)  # Source node at centre
    connector_nodes = []
    demand_nodes = []
    available_node_loctions = [(x, y) for x in range(20) for y in range(20)]
    edges = []
    
    G = nx.DiGraph()

    # Generate source node at (0,0)
    G.add_node(source_node, demand=0, source=True)
    available_node_loctions.remove(source_node)

    # Generate connector nodes at random locations in the grid near the source node
    for i in range(num_connect):
        # Choose a random location within a 5x5 area around the source node
        x = random.randint(max(0, source_node[0] - 5), min(19, source_node[0] + 5))
        y = random.randint(max(0, source_node[1] - 5), min(19, source_node[1] + 5))
        node = (x, y)
        G.add_node(node, demand=0, source=False)
        connector_nodes.append(node)
        available_node_loctions.remove(node)

    # Modify demand node generation to prioritize edge locations
    edge_locations = [(x, 0) for x in range(20)] + [(x, 19) for x in range(20)] + [(0, y) for y in range(20)] + [(19, y) for y in range(20)]
    edge_locations = list(set(edge_locations) - {source_node})  # Exclude the source node location
    for i in range(num_demand):
        node = random.choice(edge_locations)
        G.add_node(node, demand=random.choice(demands), source=False)
        demand_nodes.append(node)
        edge_locations.remove(node)

    # Connect source node to its nearest connector node
    distances = [(node, np.linalg.norm(np.array(node) - np.array(source_node))) for node in connector_nodes]
    nearest_connector = min(distances, key=lambda x: x[1])[0]
    G.add_edge(source_node, nearest_connector, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
    edges.append((source_node, nearest_connector))

    # Connect each connector node to its nearest neighboring connector node in a directed manner
    unconnected_nodes = set(connector_nodes)
    connected_nodes = {nearest_connector}

    while unconnected_nodes:
        current_node = connected_nodes.pop()
        distances = [(node, np.linalg.norm(np.array(node) - np.array(current_node))) for node in unconnected_nodes if node != current_node]
        if distances:
            nearest_node = min(distances, key=lambda x: x[1])[0]
            G.add_edge(current_node, nearest_node, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
            edges.append((current_node, nearest_node))
            connected_nodes.add(nearest_node)
            unconnected_nodes.remove(nearest_node)

    for demand_node in demand_nodes:
        distances = [(node, np.linalg.norm(np.array(node) - np.array(demand_node))) for node in connector_nodes]
        if distances:
            nearest_connector = min(distances, key=lambda x: x[1])[0]
            G.add_edge(nearest_connector, demand_node, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
            edges.append((nearest_connector, demand_node))

    # Add redundant pipes between connector nodes
    for i in range(redundant_pipes):
            node1, node2 = random.sample([source_node] + connector_nodes, 2)
            # Ensure the two nodes are not already connected and are not the same node
            while G.has_edge(node1, node2) or G.has_edge(node2, node1) or node1 == node2:
                node1, node2 = random.sample(connector_nodes, 2)
            G.add_edge(node1, node2, diameter=random.choice(pipe_diameters), roughness=random.choice(pipe_roughnesses))
            edges.append((node1, node2))

    return G, edges, connector_nodes, demand_nodes

# Visualise the network
def display_network(G, edges, connector_nodes, demand_nodes):
    
    # Filter nodes to only include source, demand, and connector nodes
    source_node = [(10, 10)]  # Source node is always at (0, 0)
    visible_nodes = set(source_node + connector_nodes + demand_nodes)
    
    # Update positions and labels to only include visible nodes
    pos = {node: node for node in visible_nodes}
    labels = {node: str(node) for node in visible_nodes}
    
    # Draw the filtered network
    nx.draw(G.subgraph(visible_nodes), pos, with_labels=True, labels=labels, node_size=500, node_color='lightblue', font_size=8)
    nx.draw_networkx_nodes(G, pos, nodelist=connector_nodes, node_color='orange', label='Connector Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=demand_nodes, node_color='red', label='Demand Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=source_node, node_color='green', label='Source Node')
    plt.legend()
    plt.show()

# Run File
G, edges, connector_nodes, demand_nodes = generate_network(num_connect, num_demand, pipe_diameters, pipe_roughnesses, redundant_pipes, demands)
display_network(G, edges, connector_nodes, demand_nodes)