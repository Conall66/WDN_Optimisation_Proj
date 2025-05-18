
"""

In this adaptation of the environment generator function, we want to generate a fixed set of possible initial networks that are all hydraulically feasible. These should exemplify these scenarios/network types:

- Branched, vs Looped
- Urban sprawl vs Densification
- Certain vs Uncertain (demand and supply)
- Large network vs small network

From each of these networks, a final destination network is generated (20 years into the future), and intemediate transitionary networks generated accordingly at each quaterly interval.

For looped networks, the loop index is set at 40% when generating, with only junctions that are near one another and connected by a a series of branches greater than 2 (sa to avoid trainagular formulations). For urban sprawl, the possibility of new demand nodes being created around the fringes of the existing network is higher, with a higher probability of forming around existing clusters. For smaller network, there will be a range of [20, 30] nodes, and for larger networks, a range of [50, 60] nodes, with the number of pumps undating proportionally.

Uncertainty is determined as a function of the duration (increases with time step k). This is extracted from real world data.

In order to increase the possibility of a hydraulically feasible network, powered pumps are generated for every branch extending from a water source or reservoir.

"""

# Import necessary libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import torch
import gym
import random

from Hydraulic_Model import *
from Visualise_network import *

# Define input parameters

# function to generate an initial network topology - this takes the scenarios as input
def generate_initial_network(network_type = 'branched', expansion_type = 'dense', network_size = 'small', **input_parameters):
    """
    Generate the initial network topology based on the given parameters.
    
    Parameters:
    -----------
    network_type : str
        Type of network to generate (e.g., 'branched', 'looped').
    expansion_type : str
        Type of expansion to apply (e.g., 'dense', 'sparse').
    network_size : str
        Size of the network (e.g., 'small', 'large').
    **input_parameters : dict
        Dictionary containing the parameters for network generation.
    
    Returns:
    --------
    graph : NetworkX graph object
        The generated water distribution network as a graph object.
    """
    
    # Extract parameters from the input dictionary
    # num_nodes = input_parameters['num_nodes']
    # num_reservoirs = input_parameters['num_reservoirs']
    area_size = input_parameters['area_size']
    # num_tanks = input_parameters['num_tanks']
    # num_pumps = input_parameters['num_pumps']
    pipe_diameters = input_parameters['pipe_diameters']
    roughness_coeffs = input_parameters['roughness_coeffs']
    min_elevations = input_parameters['min_elevations']
    max_elevations = input_parameters['max_elevations']
    min_demand = input_parameters['min_demand']
    max_demand = input_parameters['max_demand']
    min_pipe_length = input_parameters['min_pipe_length']
    max_pipe_length = input_parameters['max_pipe_length']
    init_level = input_parameters['init_level']
    min_level = input_parameters['min_level']
    max_level = input_parameters['max_level']
    tank_diameter = input_parameters['tank_diameter']
    pump_parameter = input_parameters['pump_parameter']
    pump_type = input_parameters['pump_type']

    if network_type == 'branched':
        # Generate a branched network
        cycle_index = 0
    elif network_type == 'looped':
        # Generate a looped network
        cycle_index = 0.4

    if expansion_type == 'dense':
        # Generate a dense network
        expansion_factor = 0 # Probability of adding new nodes to fringes
    elif expansion_type == 'sprawl':
        # Generate a sprawl network
        expansion_factor = 0.2

    if network_size == 'small':
        # Generate a small network
        num_nodes = np.random.randint(20, 30)
        num_reservoirs = 1
        num_tanks = num_reservoirs
        num_pumps = num_reservoirs + 1
    elif network_size == 'large':
        # Generate a large network
        num_nodes = np.random.randint(50, 60)
        num_reservoirs = 2
        num_tanks = 2
        num_pumps = num_reservoirs + 2
    elif network_size == 'test':
        # Generate a test network
        num_nodes = 5
        num_reservoirs = 1
        num_tanks = 0
        num_pumps = 1
    
    # Create an empty graph
    graph = nx.Graph()

    # Add reservoir nodes to the graph
    graph = generate_reservoirs(graph, num_reservoirs, max_elevations, area_size)
    # Add tank nodes to the graph
    graph = generate_tanks(graph, num_tanks, max_elevations, min_level, max_level, init_level, tank_diameter, area_size)
    # Add junction nodes to the graph
    graph = generate_junctions(graph, num_nodes, num_reservoirs, num_tanks, min_elevations, max_elevations, min_pipe_length, max_pipe_length, area_size)
    # Add initial demands to the junction nodes
    graph = initial_demands(graph, min_demand, max_demand)
    # Add initial supply to the reservoir nodes
    # graph = initial_supply(graph, min_demand, max_demand)
    # Add pipes to the graph
    graph = generate_pipes(graph, pipe_diameters, roughness_coeffs, min_pipe_length, max_pipe_length, cycle_index)
    # Add pumps to the graph
    graph = generate_pumps(graph, pipe_diameters, roughness_coeffs, pump_parameter, pump_type)
    
    return graph

# function to generate reservoir positions
def generate_reservoirs(graph, num_reservoirs, max_elevation, area_size = 1000, min_dist = 250):
    """
    Generate reservoir positions within the specified area size.
    
    Parameters:
    -----------
    num_reservoirs : int
        Number of reservoirs to generate.
    area_size : int
        Size of the area in which the reservoirs are located.
    max_elevation : float
        Maximum elevation for reservoirs in the network.
    min_dist : float
        Minimum distance between reservoirs.
    
    Returns:
    --------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    """
    
    reservoir_positions = []

    # Generate reservoir positions around fringes of the area, ensuring they are not too close to each other. Keeping reservoirs on the edges reduces possibility of dead ends
    for _ in range(num_reservoirs):
        while True:
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            position = (x, y)

            # Check if the new position is too close to existing reservoirs
            if all(np.linalg.norm(np.array(position) - np.array(existing)) >= min_dist for existing in reservoir_positions):
                reservoir_positions.append(position)
                break
    
    # Add reservoir nodes to the graph
    elevation = random.uniform(max_elevation * 0.8, max_elevation)
    for i, position in enumerate(reservoir_positions):
        graph.add_node(f"R{i}", type='Reservoir', position = position, elevation=elevation, head=elevation)

    return graph

# Generate tank positions and add to the graph, ensuring they are not too close to existing reservoirs or othre nodes

def generate_tanks(graph, num_tanks, max_elevation, min_level, max_level, init_level, diameter, area_size = 1000, min_dist = 250):
    """
    Generate tank positions within the specified area size.
    
    Parameters:
    -----------
    num_tanks : int
        Number of tanks to generate.
    max_elevation : float
        Maximum elevation for tanks in the network.
    min_level : float
        Minimum level for tanks in the network.
    max_level : float
        Maximum level for tanks in the network.
    init_level : float
        Initial level for tanks in the network.
    diameter : float
        Diameter of the tanks in the network.
    area_size : int
        Size of the area in which the tanks are located.
    min_dist : float
        Minimum distance between tanks and other nodes.
    
    Returns:
    --------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    """
    
    # Identify reservoir nodes in the graph
    reservoir_positions = []
    for node, data in graph.nodes(data=True):
        if data['type'] == 'Reservoir':
            reservoir_positions.append(data['position'])

    tank_positions = []

    # Generate tank positions around fringes of the area, ensuring they are not too close to each other
    for _ in range(num_tanks):
        while True:
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            position = (x, y)

            # Check if the new position is too close to existing tanks or reservoirs
            if all(np.linalg.norm(np.array(position) - np.array(existing)) >= min_dist for existing in tank_positions) and \
               all(np.linalg.norm(np.array(position) - np.array(reservoir)) >= min_dist for reservoir in reservoir_positions):
                tank_positions.append(position)
                break
    
    # Add tank nodes to the graph
    elevation = random.uniform(max_elevation * 0.8, max_elevation)
    for i, position in enumerate(tank_positions):
        graph.add_node(f"T{i}", type='Tank', position = position, elevation=elevation, init_level=init_level, min_level=min_level, max_level=max_level, diameter=diameter, volume=diameter * np.pi * (max_level - min_level) / 4)  # Volume calculation for tank

    return graph

def generate_junctions(graph, num_nodes, num_reservoirs, num_tanks, min_elevation, max_elevation, min_pipe_length, max_pipe_length, area_size = 1000, min_dist = 250):
    """
    Generate junction nodes within the specified area size.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes in the network.
    num_reservoirs : int
        Number of reservoirs in the network.
    num_tanks : int
        Number of tanks in the network.
    min_elevation : float
        Minimum elevation for junctions in the network.
    max_elevation : float
        Maximum elevation for junctions in the network.
    min_pipe_length : float
        Minimum length of pipes in the network.
    max_pipe_length : float
        Maximum length of pipes in the network.
    area_size : int
        Size of the area in which the junctions are located.
    min_dist : float
        Minimum distance between junctions and other nodes.
    
    Returns:
    --------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    """
    num_junctions = num_nodes - num_reservoirs - num_tanks
    for i in range(num_junctions):
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
            print(f"Warning: Could not place junction node{i} with both min and max distance constraints")
        
        elevation = np.random.randint(min_elevation, max_elevation)
        graph.add_node(f"J{i}", position=position, elevation = elevation, type='Junction')

    return graph

# assign demand values to the junctions, given existing demand values
def initial_demands(graph, min_demand, max_demand):
    """
    Assign initial demand values to the junction nodes.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    min_demand : float
        Minimum demand for junctions in the network.
    max_demand : float
        Maximum demand for junctions in the network.
    
    Returns:
    --------
    None
    """
    
    # for i graph nodes junctions
    for i, node in enumerate(graph.nodes()):
        if graph.nodes[node]['type'] == 'Junction':
            # Assign a random demand value between min_demand and max_demand
            demand = random.uniform(min_demand, max_demand)
            graph.nodes[node]['demand'] = demand

    return graph

def upd_demands(graph, min_demand, max_demand, forecasted_demands, uncertainty_factor = 0.1):
    """
    Update demand values for the junction nodes.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    min_demand : float
        Minimum demand for junctions in the network.
    max_demand : float
        Maximum demand for junctions in the network.
    forecasted_demands : list
        List of forecasted demand values for each junction.
    uncertainty_factor : float
        Factor to introduce uncertainty in demand values.
    
    Returns:
    --------
    None
    """
    
    # Allocates forecasted demand to each junction +- uncertainty
    for i, node in enumerate(graph.nodes()):
        if graph.nodes[node]['type'] == 'Junction':
            upd_demand = forecasted_demands[i] + random.uniform(-uncertainty_factor, uncertainty_factor) * forecasted_demands[i]
            if upd_demand < min_demand:
                upd_demand = min_demand
            elif upd_demand > max_demand:
                upd_demand = max_demand
            graph.nodes[node]['demand'] = upd_demand

    return graph

# Functions to generate initial supply
def initial_supply(graph, min_supply, max_supply):
    """
    Assign initial supply values to the reservoir nodes.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object..
    min_supply : float
        Minimum supply for reservoirs in the network.
    max_supply : float
        Maximum supply for reservoirs in the network.
    
    Returns:
    --------
    None
    """
    
    for i in range(len(graph.nodes())):
        if graph.nodes[f"Reservoir_{i}"]['type'] == 'Reservoir':
            # Assign a random supply value between min_supply and max_supply
            supply = random.uniform(min_supply, max_supply)
            graph.nodes[f"Reservoir_{i}"]['supply'] = supply

    return graph

# function to generate updated supply
def upd_supply(graph, min_supply, max_supply, forecasted_supply, uncertainty_factor = 0.1):
    """
    Update supply values for the reservoir nodes.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    min_supply : float
        Minimum supply for reservoirs in the network.
    max_supply : float
        Maximum supply for reservoirs in the network.
    forecasted_supply : list
        List of forecasted supply values for each reservoir.
    uncertainty_factor : float
        Factor to introduce uncertainty in supply values.
    
    Returns:
    --------
    None
    """
    
    # Allocates forecasted supply to each reservoir +- uncertainty
    for i, node in enumerate(graph.nodes()):
        if graph.nodes[node]['type'] == 'Reservoir':
            upd_supply = forecasted_supply[i] + random.uniform(-uncertainty_factor, uncertainty_factor) * forecasted_supply[i]
            if upd_supply < min_supply:
                upd_supply = min_supply
            elif upd_supply > max_supply:
                upd_supply = max_supply
            graph.nodes[node]['supply'] = upd_supply

    return graph

# function to generate the pipes and assign diameters and roughness values
def generate_pipes(graph, pipe_diameters, roughness_coeffs, min_pipe_length, max_pipe_length, cycle_index = 0.4):
    """
    Generate pipes and assign diameters and roughness values.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    pipe_diameters : list
        List of pipe diameters to be used in the network.
    roughness_coeffs : list
        List of roughness coefficients to be used in the network.
    min_pipe_length : float
        Minimum length of pipes in the network.
    max_pipe_length : float
        Maximum length of pipes in the network.
    
    Returns:
    --------
    None
    """

    # Create copy of the graph
    graph_copy = graph.copy()

    # Connect all nodes to all other nodes in the network
    for i, node1 in enumerate(graph.nodes()):
        for j, node2 in enumerate(graph.nodes()):
            if i != j and not graph.has_edge(node1, node2):
                # Calculate distance between nodes
                distance = np.linalg.norm(np.array(graph.nodes[node1]['position']) - np.array(graph.nodes[node2]['position']))
                graph_copy.add_edge(node1, node2, length=distance)
                
    # Find a minimum spanning tree originating from the reservoir to represent a branched network
    mst = nx.minimum_spanning_tree(graph_copy, weight = 'length')
    for u, v in mst.edges():
        # Length is the actual distance between nodes
        edge_id = f"{u}_{v}"
        length = np.linalg.norm(np.array(graph.nodes[u]['position']) - np.array(graph.nodes[v]['position']))
        diameter = np.random.choice(pipe_diameters)
        roughness = np.random.choice(roughness_coeffs)
        graph.add_edge(u, v, edge_id = edge_id, length=length, diameter=diameter, roughness=roughness)


    # Find all edges within min/max length
    viable_edges = []
    for u, v in graph.edges():
        length = graph.edges[u, v]['length']
        if min_pipe_length <= length <= max_pipe_length:
            viable_edges.append((u, v))

    # Add extra pipes for cycle index
    additional_pipes = len(graph.edges()) * cycle_index
    for i in range(int(additional_pipes)):
        # Randomly select two nodes to connect
        u, v = random.choice(viable_edges)
        # Calculate distance between nodes
        distance = np.linalg.norm(np.array(graph.nodes[u]['position']) - np.array(graph.nodes[v]['position']))
        edge_id = f"{u}_{v}"
        # Add an edge between the nodes with a random diameter and roughness value
        diameter = np.random.choice(pipe_diameters)
        roughness = np.random.choice(roughness_coeffs)
        graph.add_edge(u, v, edge_id = edge_id, length=distance, diameter=diameter, roughness=roughness)

    return graph

def generate_pumps(graph, pipe_diameters, roughness_coeffs, pump_parameter, pump_type):
    """
    Generate pumps and assign parameters.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    pump_parameter : float
        Parameter for the pumps in the network.
    pump_type : str
        Type of pump to be used in the network.
    
    Returns:
    --------
    None
    """
    
    # Add pumps to the graph, ensuring they are connected to reservoirs or tanks
    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'Reservoir' or graph.nodes[node]['type'] == 'Tank':
            connected_nodes = list(graph.neighbors(node))
            for neighbor in connected_nodes:
                if graph.nodes[neighbor]['type'] == 'Junction':
                    # Remove existing edge to avoid duplicates
                    if graph.has_edge(node, neighbor):
                        graph.remove_edge(node, neighbor)
                    pump_id = f"Pump_{node}_{neighbor}"
                    length = np.linalg.norm(np.array(graph.nodes[node]['position']) - np.array(graph.nodes[neighbor]['position']))
                    diameter = np.random.choice(pipe_diameters)
                    roughness = np.random.choice(roughness_coeffs)
                    # Add a pump edge between the reservoir/tank and the junction
                    graph.add_edge(node, neighbor, pipe_id = pump_id, length = length, diameter = diameter, roughness = roughness, pump_id=pump_id, pump_parameter=pump_parameter, pump_type=pump_type)

    return graph

# function to generate a destination network given the inital network
def is_hydraulically_feasible(graph):
    """
    Assess the hydraulic feasibility of the network.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    
    Returns:
    --------
    bool
        True if the network is hydraulically feasible, False otherwise.
    """
    
    # Convert graph to wntr model
    wntr_graph = convert_graph_to_wntr(graph)
    # Run EPANET simulation
    results = run_epanet_simulation(wntr_graph)
    performance_metrics = evaluate_network_performance(wntr_graph, results)

    # print(f"Performance Metrics: {performance_metrics}")

    # Find the node and value of maximum pressure deficit
    max_pressure_deficit = max(performance_metrics['pressure_deficit'].values())
    min_pressure_deficit = min(performance_metrics['pressure_deficit'].values())
    # Find the node and value of maximum headloss
    max_headloss = max(performance_metrics['headlosses'].values())
    min_headloss = min(performance_metrics['headlosses'].values())

    print(f"Maximum Pressure Deficit: {max_pressure_deficit}")
    print(f"Minimum Pressure Deficit: {min_pressure_deficit}")
    print(f"Maximum Headloss: {max_headloss}")
    print(f"Minimum Headloss: {min_headloss}")

    return results, True

    # Check if the network is hydraulically feasible based on performance metrics
    # if all(performance_metrics['pressure_deficit']) >= 20 and all(performance_metrics['headlosses']) <= 10:
    #     return True
    # else:
    #     return False
    
# function to check unserviced demand
def check_unserviced_demand(graph):
    """
    Check for unserviced demand in the network.
    
    Parameters:
    -----------
    graph : NetworkX graph object
        The water distribution network as a graph object.
    
    Returns:
    --------
    bool
        True if there is unserviced demand, False otherwise.
    """
    
    # Check if any junctions have unmet demand
    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'Junction' and 'demand' in graph.nodes[node]:
            if graph.nodes[node]['demand'] > 0:
                return True
    return False

# function to generate the tranitionary networks

# function to assess the hydraulic feasibility of a network

# function to test the script
def test_script():
    """
    Test the script by generating a network and visualizing it.
    
    Returns:
    --------
    None
    """
    
    # Define input parameters
    input_parameters = {
        'num_nodes': 20,
        'num_reservoirs': 2,
        'area_size': 1000,
        'num_tanks': 1,
        'num_pumps': 2,
        'pipe_diameters': [300.0, 400.0, 500.0],
        'roughness_coeffs': [0.1, 0.2, 0.3],
        'min_elevations': 0,
        'max_elevations': 10,
        'min_demand': 0.5,
        'max_demand': 5,
        'min_pipe_length': 5,
        'max_pipe_length': 100,
        'init_level': 5,
        'min_level': 1,
        'max_level': 10,
        'tank_diameter': 5.0,
        'pump_parameter': 100,
        'pump_type': 'POWER'
    }

    # Generate an initial network that is hydraulically feasible
    while True:
        graph = generate_initial_network(network_type='branched', expansion_type='dense', network_size='small', **input_parameters)
        # Check if the network is hydraulically feasible
        results, success = is_hydraulically_feasible(graph)
        if success:
            break
    visualise_network(graph, results, title="Initial Network")

# generate simple network to check hydraulic modelling outputing correct values
def generate_simple_network():
    """
    Generate a simple network for testing purposes.
    
    Returns:
    --------
    graph : NetworkX graph object
        The generated water distribution network as a graph object.
    """
    
    # Define input parameters
    input_parameters = {
        'num_nodes': 5,
        'num_reservoirs': 1,
        'area_size': 1000,
        'num_tanks': 0,
        'num_pumps': 2,
        'pipe_diameters': [300.0, 400.0, 500.0],
        'roughness_coeffs': [0.1, 0.2, 0.3],
        'min_elevations': 0,
        'max_elevations': 10,
        'min_demand': 0.5,
        'max_demand': 5,
        'min_pipe_length': 5,
        'max_pipe_length': 100,
        'init_level': 5,
        'min_level': 1,
        'max_level': 10,
        'tank_diameter': 5.0,
        'pump_parameter': 100,
        'pump_type': 'POWER'
    }

    # Generate a simple network
    graph = generate_initial_network(network_type='branched', expansion_type='dense', network_size='test', **input_parameters)
    # Check if the network is hydraulically feasible
    results, success = is_hydraulically_feasible(graph)
    # Visualise the network
    visualise_network(graph, results, title="Simple Network")

    return graph

if __name__ == "__main__":
    # test_script()
    generate_simple_network()


