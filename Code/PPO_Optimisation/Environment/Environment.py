
"""

In this file, we generate an intitial water distribution network model using networkx graphs. We encode the input parameters to determine an update of the network.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import random


def generate_initial_network(
        num_nodes, 
        num_reservoirs,
        area_size, 
        num_tanks, 
        num_pumps, 
        pipe_diameters, 
        roughness_coeffs,
        min_elevations,
        max_elevations,
        min_demand,
        max_demand,
        min_pipe_length,
        max_pipe_length,
        min_level,
        max_level, 
        diameter,
        pump_parameter,
        pump_type,
        network_type = 'branched'):

    """
    Generate an initial water distribution network model using networkx graphs.

    Parameters:
    num_nodes (int): Total number of nodes in the network.
    num_reservoirs (int): Number of reservoirs in the network.
    area_size (float): Size of the area in which the network is generated.
    num_tanks (int): Number of tanks in the network.
    num_pumps (int): Number of pumps in the network.
    pipe_diameters (list): List of possible pipe diameters.
    roughness_coeffs (list): List of possible roughness coefficients for pipes.
    min_elevations (int): minimum elevations for tanks and reservoirs.
    max_elevations (int): maximum elevations for tanks.
    min_demand (float): Minimum demand for junctions.
    max_demand (float): Maximum demand for junctions.
    min_pipe_length (float): Minimum length of pipes.
    max_pipe_length (float): Maximum length of pipes.
    min_level (float): Minimum water level in tanks.
    max_level (float): Maximum water level in tanks.
    diameter (float): Diameter of pipes.
    pump_parameter (float): Parameter for the pump.
    pump_type (str): Type of pump ('Centrifugal', 'Positive Displacement', etc.).
    network_type (str): Type of network to generate ('branched' or 'looped').

    Returns:
    graph (networkx.Graph): The generated water distribution network model as a graph.
    """

    graph = nx.Graph()

    for i in range(num_reservoirs):
        node_id = f"R{i}"
        # elevation = np.random.randint(min_elevations, max_elevations)
        elevation = max_elevations # add reservoirs to highest point to simulate gravity fed system
        position = (np.random.randint(0, area_size), np.random.randint(0, area_size))
        graph.add_node(node_id, position = position, type='Reservoir', head=elevation, elevation=elevation)

    for i in range(num_tanks):
        node_id = f"T{i}"
        max_attempts = 100
        attempt_count = 0
        valid_position = False
        while attempt_count < max_attempts and not valid_position:
            position = (np.random.randint(0, area_size), np.random.randint(0, area_size))
            # Calculate distance to reservoirs
            distances = [
                np.linalg.norm(np.array(position) - np.array(graph.nodes[n]['position'])) for n in graph.nodes() if graph.nodes[n]['type'] == 'Reservoir' or graph.nodes[n]['type'] == 'Tank']
            
            # We want the tanks to be far away from the reservoirs to represent a more realistic system
            min_distance_ok = all(d > max_pipe_length for d in distances)
            
            if min_distance_ok:
                valid_position = True
            
            attempt_count += 1

        if not valid_position:
            # If we couldn't find a valid position after max attempts,
            # place node in a position that at least satisfies the minimum distance
            while True:
                position = (np.random.randint(0, area_size), np.random.randint(0, area_size))
                distances = [np.linalg.norm(np.array(position) - np.array(graph.nodes[n]['position'])) for n in graph.nodes()]
                if all(d > min_pipe_length for d in distances) or not distances:
                    break
            print(f"Warning: Could not place junction {node_id} further than 50km from reservoirs or other tanks")

        elevation = np.random.randint(min_elevations, max_elevations)
        init_level = np.random.randint(min_level, max_level)
        graph.add_node(node_id, position = position, type='Tank', elevation=elevation, init_level=init_level, min_level=min_level, max_level=max_level, diameter=diameter, volume = round(np.pi * (diameter/2)**2 * (max_level - min_level),2))

    num_junctions = num_nodes - num_reservoirs - num_tanks
    for i in range(num_junctions):
        node_id = f"J{i}"
        max_attempts = 100  # Limit the number of attempts to avoid infinite loops
        attempt_count = 0
        valid_position = False
        
        while attempt_count < max_attempts and not valid_position:
            position = (np.random.randint(0, area_size), np.random.randint(0, area_size))
            # Calculate distance to all other nodes in the graph
            distances = [np.linalg.norm(np.array(position) - np.array(graph.nodes[n]['position'])) for n in graph.nodes()]
            
            # Check if the position is within proper distance constraints:
            # 1. Not too close to any existing node (greater than min_pipe_length)
            # 2. Not too far from at least one node (less than max_pipe_length)
            min_distance_ok = all(d > min_pipe_length for d in distances)
            max_distance_ok = any(d <= max_pipe_length for d in distances) if distances else True
            
            if min_distance_ok and max_distance_ok:
                valid_position = True
            
            attempt_count += 1
        
        if not valid_position:
            # If we couldn't find a valid position after max attempts,
            # place node in a position that at least satisfies the minimum distance
            while True:
                position = (np.random.randint(0, area_size), np.random.randint(0, area_size))
                distances = [np.linalg.norm(np.array(position) - np.array(graph.nodes[n]['position'])) for n in graph.nodes()]
                if all(d > min_pipe_length for d in distances) or not distances:
                    break
            print(f"Warning: Could not place junction {node_id} with both min and max distance constraints")
        
        elevation = np.random.randint(min_elevations, max_elevations)
        demand = np.random.randint(min_demand, max_demand)
        graph.add_node(node_id, position=position, elevation = elevation, type='Junction', demand=demand)

    # Create a copy of the graph to avoid modifying the original graph
    graph_copy = graph.copy()
    for i, node1 in enumerate(graph_copy.nodes()):
        for node2 in list(graph_copy.nodes())[i+1:]:
            # Calculate distance between nodes
            if graph_copy.has_edge(node1, node2) or graph.has_edge(node2, node1) or node1 == node2:
                continue
            pos1 = graph_copy.nodes[node1]['position']
            pos2 = graph_copy.nodes[node2]['position']
            length = np.linalg.norm(np.array(pos1) - np.array(pos2))
            # Add edge with weight based on distance
            graph_copy.add_edge(node1, node2, length=length)

    # Create a minimum spanning tree to connect all nodes
    mst = nx.minimum_spanning_tree(graph_copy, weight = 'length')
    for u, v in mst.edges():
        # Length is the actual distance between nodes
        edge_id = f"{u}_{v}"
        length = np.linalg.norm(np.array(graph.nodes[u]['position']) - np.array(graph.nodes[v]['position']))
        diameter = np.random.choice(pipe_diameters)
        roughness = np.random.choice(roughness_coeffs)
        graph.add_edge(u, v, edge_id = edge_id, length=length, diameter=diameter, roughness=roughness)

    # print(f"Generated {len(graph.nodes())} nodes and {len(graph.edges())} edges in the network.")
    # print(f"Graph Nodes: {graph.nodes()}")
    # print(f"Graph Edges: {graph.edges()}")

    # Replace the edge connecting the reservoir to the next junction with a pumped edge: once for each reservoir
    # for i in range(num_reservoirs):
    #     # identify a node connectde to the reservoir
    #     reservoir_node = f"R{i}"
    #     connected_nodes = list(graph.neighbors(reservoir_node))
    #     if connected_nodes:
    #         # Select the first connected node
    #         connected_node = connected_nodes[0]
    #         # Remove the existing edge
    #         graph.remove_edge(reservoir_node, connected_node)
    #         # Calulcate the length of the new edge
    #         length = np.linalg.norm(np.array(graph.nodes[reservoir_node]['position']) - np.array(graph.nodes[connected_node]['position']))
    #         diameter = np.random.choice(pipe_diameters)
    #         roughness = np.random.choice(roughness_coeffs)
    #         # Add a new pump edge
    #         graph.add_edge(reservoir_node, connected_node, length = length, diameter = diameter, roughness = roughness, pump_parameter=pump_parameter, pump_type=pump_type)

    # Add pumps to the network for every edge coming from a reservoir
    for i in range(num_reservoirs):
        # identify a node connected to the reservoir
        reservoir_node = f"R{i}"
        connected_nodes = list(graph.neighbors(reservoir_node))
        for connected_node in connected_nodes:
        # Remove the existing edge
            graph.remove_edge(reservoir_node, connected_node)
            # Calculate the length of the new edge
            length = np.linalg.norm(np.array(graph.nodes[reservoir_node]['position']) - np.array(graph.nodes[connected_node]['position']))
            diameter = np.random.choice(pipe_diameters)
            roughness = np.random.choice(roughness_coeffs)
            # Add a new pump edge
            graph.add_edge(reservoir_node, connected_node, length=length, diameter=diameter, roughness=roughness, pump_parameter=pump_parameter, pump_type=pump_type)
            
    candidate_edges = []
    # if network_type == 'looped', add additional edges within min/max length range
    if network_type == 'looped':
        # Get all junction nodes
        junction_nodes = [node for node, attr in graph.nodes(data=True) if attr['type'] == 'Junction']
        
        # Identify any unconnected junctions within the distance range
        for i in range(len(junction_nodes)):
            for j in range(i + 1, len(junction_nodes)):
                node_i = junction_nodes[i]
                node_j = junction_nodes[j]
                if not graph.has_edge(node_i, node_j) and not graph.has_edge(node_j, node_i):
                    # Calculate the distance between the nodes
                    length = np.linalg.norm(np.array(graph.nodes[node_i]['position']) - np.array(graph.nodes[node_j]['position']))
                    if min_pipe_length <= length <= max_pipe_length:
                        # Add the edge to candidate list
                        candidate_edges.append((node_i, node_j, length))

        # Correctly iterate through the list of candidate edges
        # for node_i, node_j, length in candidate_edges:
        #     if np.random.random() < 0.4:  # 40% chance to add the edge
        #         # Add the edge with a random diameter and roughness
        #         edge_id = f"{node_i}_{node_j}"
        #         diameter = np.random.choice(pipe_diameters)
        #         roughness = np.random.choice(roughness_coeffs)
        #         graph.add_edge(node_i, node_j, edge_id=edge_id, length=length, diameter=diameter, roughness=roughness)

        # Multiply the existing nuber of edges by 40% and randomly select that number of edges to add from the candidate list
        num_edges_to_add = int(len(graph.edges()) * 0.4)
        while candidate_edges and num_edges_to_add:
            selected_edge = random.choice(candidate_edges)
            node_i, node_j, length = selected_edge
            # Add the edge with a random diameter and roughness
            edge_id = f"{node_i}_{node_j}"
            diameter = np.random.choice(pipe_diameters)
            roughness = np.random.choice(roughness_coeffs)
            graph.add_edge(node_i, node_j, edge_id=edge_id, length=length, diameter=diameter, roughness=roughness)
            candidate_edges.remove(selected_edge)

            num_edges_to_add -= 1

    print(f"Generated {len(graph.nodes())} nodes and {len(graph.edges())} edges in the network.")

    return graph

# Design an end network given the initial network with stochastic demand probabilities

# Design a staged-design programme to extract networks for each time step
    

# Visualise the network
def draw_network(graph):
    """
    Draw the water distribution network.

    Parameters:
    graph (networkx.Graph): The water distribution network graph.

    Returns:
    None
    """

    # print(f"Graph Nodes: {graph.nodes()}")

    plt.figure(figsize=(10, 10))
    plt.title("Water Distribution Network")
    # Reservoir nodes are blue circles, tanks are green squares, junctions are red triangles, pumps are purple diamonds
    node_colors = []
    node_shapes = []
    for node, data in graph.nodes(data=True):
        if data['type'] == 'Reservoir':
            node_colors.append('blue')
            node_shapes.append('o')
        elif data['type'] == 'Tank':
            node_colors.append('green')
            node_shapes.append('s')
        elif data['type'] == 'Junction':
            node_colors.append('red')
            node_shapes.append('^')
        # elif data['type'] == 'Pump':
        #     node_colors.append('purple')
        #     node_shapes.append('D')
    
    # Draw normal edges and pump edges differently
    pos = nx.get_node_attributes(graph, 'position')
    
    # First draw all nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_shape='o')
    nx.draw_networkx_labels(graph, pos)
    
    # Draw normal pipes (edges without pump attributes)
    normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' not in d]
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, edge_color='gray')
    
    # Draw pump edges in a different color
    pump_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' in d]
    nx.draw_networkx_edges(graph, pos, edgelist=pump_edges, edge_color='purple', width=2.0)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(graph, 'edge_id')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_weight='bold')
    
    # Add legend
    legend_labels = {
        'Reservoir': 'blue',
        'Tank': 'green',
        'Junction': 'red',
        'Pump': 'purple'
    }
    for label, color in legend_labels.items():
        plt.scatter([], [], color=color, label=label, s=100)
    plt.legend()

    plt.show()

# Test the function
if __name__ == "__main__":
    num_nodes = 50
    num_reservoirs = 2
    area_size = 1000
    num_tanks = 2
    num_pumps = 2
    pipe_diameters = [0.1, 0.15, 0.2]
    roughness_coeffs = [0.01, 0.02, 0.03]
    min_elevations = 10
    max_elevations = 30
    min_demand = 5
    max_demand = 20
    min_pipe_length = 25
    max_pipe_length = 200
    min_level = 5
    max_level = 15
    diameter = 0.1
    pump_parameter = 1.0
    pump_type = 'POWER',
    network_type = 'looped'

    graph = generate_initial_network(num_nodes, num_reservoirs, area_size, num_tanks, num_pumps, pipe_diameters, roughness_coeffs, min_elevations, max_elevations, min_demand, max_demand, min_pipe_length, max_pipe_length, min_level, max_level, diameter, pump_parameter, pump_type, network_type)

    draw_network(graph)
    
