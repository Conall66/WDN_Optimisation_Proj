
"""

In this function, we generate an initial branched network given input parameters

"""

# Importing libraries
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import wntr
import pandas as pd
import os
import networkx as nx

from Elevation_map import generate_elevation_map
from Demand_profiles import *
from Visualise_2 import visualise_wntr
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

def generate_branched_network(
        pipes, # A dictionary containing the pipe data
        roughness, # A dictionary containing the roughness data
        area_size = (100, 100),
        elevation_range = (0, 100),
        num_junctions = 10,
        num_reservoirs = 1,
        num_tanks = 1,
        num_pumps = 1,
        landscape = 'flat',
        environment = 'urban',
        init_level = 0.5,
        min_level = 0.2,
        max_level = 0.8,
        tank_diameter = 0.5,
        ):
    
    """
    Generate a branched water distribution network with given parameters.
    
    Parameters:
    -----------
    pipes : dict
        Dictionary containing pipe data (length, diameter, roughness)
    roughness : dict
        Dictionary containing roughness data for each pipe
    elevation_map : np.array
        2D numpy array representing the elevation of the landscape
    area_size : list
        Size of the area in which to generate the network
    num_junctions : int
        Number of junctions in the network
    num_reservoirs : int
        Number of reservoirs in the network
    num_tanks : int
        Number of tanks in the network
    num_pumps : int
        Number of pumps in the network
    landscape : str
        Type of landscape ('flat', 'hilly', 'mountainous')
    environment : str
        Type of environment ('urban', 'rural')
    
    Returns:
    --------
    wn : WNTR model object
        The generated water distribution network model
    
    """

    # Create a new WNTR model
    wn = wntr.network.WaterNetworkModel()

    # Generate a random elevation map
    elevation_map, peak_data = generate_elevation_map(
        area_size = area_size, 
        elevation_range = elevation_range,
        num_peaks = 5, # hard coded for now 
        landscape_type = landscape)

    # Generate demand data for junctions
    # Determine distribution of junction nodes
    if environment == 'urban':
        # More commercial nodes
        num_commercial = int(num_junctions * 0.8)
        num_residential = int(num_junctions * 0.2)
    elif environment == 'rural':
        # More residential nodes
        num_commercial = int(num_junctions * 0.2)
        num_residential = int(num_junctions * 0.8)

    # Generate demand profiles
    quarters, populations = generate_sample_growth(start_year = 2025, end_year = 2050, start_population = 20000, end_population = 50000, num_samples = 100)
    residential_demand, commercial_demand = generate_demand_profiles(populations, uncertainty = None)

    # Extract only residential demand for Y2025Q1
    residential_demand = residential_demand['Y2025Q1']
    commercial_demand = commercial_demand['Y2025Q1']

    # Extract index of the maximum residential and commercial demand
    max_residential_index = np.argmax(residential_demand)
    max_commercial_index = np.argmax(commercial_demand)
    # Extract the maximum residential and commercial demand values
    max_residential_demand = residential_demand['Demand'][max_residential_index]
    max_commercial_demand = commercial_demand['Demand'][max_commercial_index]
    # Extract uncertainty
    res_uncertainty = residential_demand['Uncertainty']
    com_uncertainty = commercial_demand['Uncertainty']

    # Add junctions with demand profiles
    for i in range(num_commercial):
        node_positions = wn.query_node_attribute('coordinates')
        while True:
            x = random.randint(0, area_size[0] - 1)
            y = random.randint(0, area_size[1] - 1)
            if (x, y) not in node_positions:
                break
        # Find elevation form elevation map
        elevation = elevation_map[x, y]
        demand = max_commercial_demand * (1 + random.uniform(-com_uncertainty, com_uncertainty))
        wn.add_junction(name=f"Commercial_{i}", base_demand=demand, coordinates=(x, y), elevation=elevation)

    for i in range(num_residential):
        # Randomly generate coordinates and check they are not already used
        node_positions = wn.query_node_attribute('coordinates')
        while True:
            x = random.randint(0, area_size[0] - 1)
            y = random.randint(0, area_size[1] - 1)
            if (x, y) not in node_positions:
                break
        # Find elevation form elevation map
        elevation = elevation_map[x, y]
        demand = max_residential_demand * (1 + random.uniform(-res_uncertainty, res_uncertainty))
        wn.add_junction(name=f"Residential_{i}", base_demand=demand, coordinates=(x, y), elevation=elevation)

    # Create a minimum spanning tree for the nodes in the wntr model
    G = wn.get_graph()
    G = G.to_undirected() # Allows for minimum spanning tree
    pos = wn.query_node_attribute('coordinates')
    nodes = list(G.nodes())

    # Connect all nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Calculate distance between nodes
            dist = math.sqrt((pos[nodes[i]][0] - pos[nodes[j]][0])**2 + (pos[nodes[i]][1] - pos[nodes[j]][1])**2)
            pipe_name = f"Pipe_{i}_{j}"
            G.add_edge(nodes[i], nodes[j], length=dist, name=pipe_name)

    # Create a minimum spanning tree
    mst = nx.minimum_spanning_tree(G, weight = 'length')

    # Display the minimum spanning tree
    # plt.figure(figsize=(10, 10))
    # nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    # plt.title('Minimum Spanning Tree of the Network')
    # plt.show()

    # Add pipes to the WNTR model
    pipe_counter = 1
    for u, v in mst.edges():
        pipe_name = f"Pipe_{pipe_counter}"
        # Extract length from the graph
        length = math.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
        pipe = random.choice(list(pipes.keys()))
        diameter = pipes[pipe]['diameter'] * 1000 # Convert diameter from m to mm
        roughness_value = random.choice(roughness)
        wn.add_pipe(name=pipe_name, start_node_name=u, end_node_name=v, length=length, diameter=diameter, roughness=roughness_value)
        pipe_counter += 1

    connected_to_reservoir = []
    # For reservoirs
    for i in range(num_reservoirs):
        if peak_data:
            x, y, z = random.choice(peak_data)
            wn.add_reservoir(name=f"Reservoir_{i}", base_head=z, coordinates=(x, y))
            # Connect to closest junction
            shortest_distance = float('inf')
            junctions = wn.junction_name_list
            for j in junctions:
                # Calculate distance to each junction
                dist = np.linalg.norm(np.array([x, y]) - np.array(wn.get_node(j).coordinates))
                if dist < shortest_distance:
                    closest_junction = j
                    shortest_distance = dist
            length = shortest_distance
            pipe_name = f"Pipe_Reservoir_{i}"
            pipe = random.choice(list(pipes.keys()))
            diameter = pipes[pipe]['diameter'] * 1000
            roughness_value = random.choice(roughness)
            wn.add_pipe(name=pipe_name, start_node_name=f"Reservoir_{i}", end_node_name=closest_junction, 
                    length=length, diameter=diameter, roughness=roughness_value)
            
            peak_data.remove((x, y, z))
            connected_to_reservoir.append(closest_junction)

    # For tanks
    for i in range(num_tanks):
        if peak_data:
            x, y, z = random.choice(peak_data)
            wn.add_tank(name=f"Tank_{i}", elevation=z, init_level=init_level, 
                    min_level=min_level, max_level=max_level, diameter=tank_diameter, coordinates=(x, y))
            # Find closest junction to connect to

            shortest_distance = float('inf')
            junctions = wn.junction_name_list
            for j in junctions:
                # Calculate distance to each junction
                dist = np.linalg.norm(np.array([x, y]) - np.array(wn.get_node(j).coordinates))
                if dist < shortest_distance:
                    closest_junction = j
                    shortest_distance = dist

            length = shortest_distance
            pipe_name = f"Pipe_Tank_{i}"
            pipe = random.choice(list(pipes.keys()))
            diameter = pipes[pipe]['diameter'] * 1000
            roughness_value = random.choice(roughness)
            wn.add_pipe(name=pipe_name, start_node_name=f"Tank_{i}", end_node_name=closest_junction, 
                    length=length, diameter=diameter, roughness=roughness_value)

    # Add pumps to the WNTR model from the reservoir to its nearest junction
    for i in range(num_reservoirs):
        reservoir = wn.get_node(f"Reservoir_{i}")
        # Find nodes connected to the reservoir from the wntr model
        # Add a pump for each connected node
        for j in range(len(connected_to_reservoir)):
            pump_name = f"Pump_{i}_{j}"
            wn.add_curve('pump_curve', 'HEAD', [[0, 300], [50, 240], [100, 200]]) # Sample values for pump curve showing that at 0 flow, the head is 100m and at 100 flow, the head is 60m - how much energy is transfered to the water given flow
            wn.add_pump(name=pump_name, start_node_name=reservoir.name, end_node_name=connected_to_reservoir[j], pump_type='HEAD', pump_parameter='pump_curve')

    return wn, elevation_map

# Test the function
if __name__ == "__main__":
    # Define parameters
    Pipes = {
    'Pipe 1': {'diameter': 0.152, 'unit_cost': 68, 'carbon_emissions': 0.48},
    'Pipe 2': {'diameter': 0.203, 'unit_cost': 91, 'carbon_emissions': 0.59},
    'Pipe 3': {'diameter': 0.254, 'unit_cost': 113, 'carbon_emissions': 0.71},
    'Pipe 4': {'diameter': 0.305, 'unit_cost': 138, 'carbon_emissions': 0.81},
    'Pipe 5': {'diameter': 0.356, 'unit_cost': 164, 'carbon_emissions': 0.87},
    'Pipe 6': {'diameter': 0.406, 'unit_cost': 192, 'carbon_emissions': 0.96},
    'Pipe 7': {'diameter': 0.457, 'unit_cost': 219, 'carbon_emissions': 1.05},
    'Pipe 8': {'diameter': 0.508, 'unit_cost': 248, 'carbon_emissions': 1.14},
    'Pipe 9': {'diameter': 0.610, 'unit_cost': 305, 'carbon_emissions': 1.32}
    }

    roughness = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005] # Example roughness values

    # Generate the network
    wn, elevation_map = generate_branched_network(Pipes, roughness)

    # Run the EPANET simulation
    results = run_epanet_simulation(wn, static = True)

    # Evaluate the network performance
    performance_metrics = evaluate_network_performance(wn, results)

    # Visualise the network
    visualise_wntr(wn, elevation_map, results, title="Branched Water Distribution Network")
    
    # Visualize the generated network
    # wntr.graphics.plot_network(wn)
    # plt.show()