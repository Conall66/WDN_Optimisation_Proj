

"""

In this file, we generate an initial branched network and chekc it's hydraulic feasibility. This takes as inputs the network size, the network type and the uncertainty level to generate initial sets of networks with initial supply and demand values. These environment will be used as the starting points for training specific and general RL agents.

"""

# import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import math
import random

from Elevation_map import generate_elevation_map
from Demand_profiles import *
from Visualise_network import visualise_network
from Hydraulic_Model import *

def generated_branched_network(
        area_size,
        num_nodes, 
        num_reservoirs, 
        num_tanks,
        initial_level,
        min_level,
        max_level,
        tank_diameter,
        num_pumps,
        pump_type,
        pump_power, 
        roughness_coefficients, 
        base_demand, 
        pipes,
        elevation_range = [0, 100],
        environment_type = 'urban',
        landscape_type = 'flat'):
    
    # Create empty graph
    G = nx.Graph()
    
    # Generate elevation map
    elevation_map, peak_data = generate_elevation_map(
        area_size=area_size,
        elevation_range=elevation_range,
        num_peaks=10,
        landscape_type=landscape_type
    )
    
    # Determine distribution of junction nodes
    num_junctions = num_nodes - num_reservoirs - num_tanks
    if environment_type == 'urban':
        # More commercial nodes
        num_commercial = int(num_junctions * 0.8)
        num_residential = int(num_junctions * 0.2)
    elif environment_type == 'rural':
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

    # Create commercial junction nodes
    for i in range(num_commercial):
        x = random.uniform(0, area_size[0])
        y = random.uniform(0, area_size[1])
        # Get elevation value from elevation map
        z = elevation_map[int(x), int(y)]
        demand = max_commercial_demand * random.uniform(1 - com_uncertainty, 1 + com_uncertainty)
        G.add_node(f"Commercial_Junction_{i}", elevation=z, demand=demand, type='commercial', coordinates=(x, y))

    # Create residential junction nodes
    for i in range(num_residential):
        x = random.uniform(0, area_size[0])
        y = random.uniform(0, area_size[1])
        # Get elevation value from elevation map
        z = elevation_map[int(x), int(y)]
        demand = max_residential_demand * random.uniform(1 - res_uncertainty, 1 + res_uncertainty)
        G.add_node(f"Residential_Junction_{i}", elevation=z, demand=demand, type='residential', coordinates=(x, y))

    # Create a complete graph with all junctions to find the minimum spanning tree
    G_copy = G.copy()
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            # Calculate distance between nodes
            distance = np.linalg.norm(np.array(G.nodes[node1]['coordinates']) - np.array(G.nodes[node2]['coordinates']))
            G_copy.add_edge(node1, node2, length=distance)
            
    # Find a minimum spanning tree for the junctions
    mst = nx.minimum_spanning_tree(G_copy, weight='length')
    
    # Add edges from MST to the original graph
    for u, v in mst.edges():
        edge_id = f"{u}_{v}"
        length = np.linalg.norm(np.array(G.nodes[u]['coordinates']) - np.array(G.nodes[v]['coordinates']))
        # Randomly select pipe from pipes
        pipe = random.choice(list(pipes.keys()))
        # Get pipe data
        pipe_data = pipes[pipe]
        diameter = pipe_data['diameter']
        unit_cost = pipe_data['unit_cost']
        carbon_emissions = pipe_data['carbon_emissions']
        roughness = random.choice(roughness_coefficients)
        # Add edge to the graph with pipe data
        G.add_edge(u, v, edge_id=edge_id, length=length, diameter=diameter, 
                  cost=unit_cost*length, carbon_emissions=carbon_emissions, roughness=roughness)
    
    # Now add reservoirs at the peaks and connect to the closest junction
    for i in range(num_reservoirs):
        if len(peak_data) > 0:
            # Select a random peak location
            x, y, z = random.choice(peak_data)
            G.add_node(f"Reservoir_{i}", elevation=z, base_head=z, type='reservoir', coordinates=(x, y))
            # Remove the selected peak location from peak data
            peak_data.remove((x, y, z))
        else:
            # If no peaks left, generate a random location
            x = random.uniform(0, area_size[0])
            y = random.uniform(0, area_size[1])
            z = random.uniform(elevation_range[0], elevation_range[1])
            G.add_node(f"Reservoir_{i}", elevation=z, base_head=z, type='reservoir', coordinates=(x, y))
        
        # Find closest junction node
        reservoir = f"Reservoir_{i}"
        min_distance = float('inf')
        closest_node = None
        for node in nodes:
            distance = np.linalg.norm(np.array(G.nodes[reservoir]['coordinates']) - np.array(G.nodes[node]['coordinates']))
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Connect reservoir to closest node with a pump
        edge_id = f"{reservoir}_{closest_node}"
        length = min_distance
        pipe = random.choice(list(pipes.keys()))
        pipe_data = pipes[pipe]
        diameter = pipe_data['diameter']
        unit_cost = pipe_data['unit_cost']
        carbon_emissions = pipe_data['carbon_emissions']
        roughness = random.choice(roughness_coefficients)
        
        # Add pump edge directly
        G.add_edge(reservoir, closest_node, edge_id=edge_id, length=length, diameter=diameter, 
                  cost=unit_cost*length, carbon_emissions=carbon_emissions, roughness=roughness,
                  pump_type=pump_type, power=pump_power)
    
    # Add tanks and connect to closest junction
    for i in range(num_tanks):
        if len(peak_data) > 0:
            # Select a random peak location
            x, y, z = random.choice(peak_data)
            G.add_node(f"Tank_{i}", elevation=z, initial_level=initial_level, min_level=min_level, 
                      max_level=max_level, diameter=tank_diameter, type='tank', coordinates=(x, y))
            # Remove the selected peak location from peak data
            peak_data.remove((x, y, z))
        else:
            # If no peaks left, generate a random location
            x = random.uniform(0, area_size[0])
            y = random.uniform(0, area_size[1])
            z = elevation_map[int(x), int(y)]
            G.add_node(f"Tank_{i}", elevation=z, initial_level=initial_level, min_level=min_level, 
                      max_level=max_level, diameter=tank_diameter, type='tank', coordinates=(x, y))
        
        # Find closest junction node
        tank = f"Tank_{i}"
        min_distance = float('inf')
        closest_node = None
        for node in nodes:
            distance = np.linalg.norm(np.array(G.nodes[tank]['coordinates']) - np.array(G.nodes[node]['coordinates']))
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Connect tank to closest node
        edge_id = f"{tank}_{closest_node}"
        length = min_distance
        pipe = random.choice(list(pipes.keys()))
        pipe_data = pipes[pipe]
        diameter = pipe_data['diameter']
        unit_cost = pipe_data['unit_cost']
        carbon_emissions = pipe_data['carbon_emissions']
        roughness = random.choice(roughness_coefficients)
        
        G.add_edge(tank, closest_node, edge_id=edge_id, length=length, diameter=diameter, 
                  cost=unit_cost*length, carbon_emissions=carbon_emissions, roughness=roughness)
    
    return G, elevation_map

# Example usage
if __name__ == "__main__":
    area_size = (100, 100)  # in km
    num_nodes = 50
    num_reservoirs = 2
    num_tanks = 2
    initial_level = 10.0  # in m
    min_level = 5.0  # in m
    max_level = 15.0  # in m
    tank_diameter = 5.0  # in m
    num_pumps = 2
    pump_type = 'POWER'
    pump_power = 10.0  # in kW
    roughness_coefficients = [0.01, 0.02, 0.03]
    base_demand = 1.0  # in L/s
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
    landscape_type = 'flat' # Flat or hilly
    elevation_range = [0, 100]
    environment_type = 'rural' # Urban or rural
    
    G, elevation_map = generated_branched_network(
        area_size,
        num_nodes,
        num_reservoirs,
        num_tanks,
        initial_level,
        min_level,
        max_level,
        tank_diameter,
        num_pumps,
        pump_type,
        pump_power,
        roughness_coefficients,
        base_demand,
        Pipes,
        elevation_range,
        environment_type,
        landscape_type
    )

    # Visualise the network
    # pos = nx.get_node_attributes(G, 'coordinates')
    # plt.contourf(elevation_map, cmap='terrain', alpha=0.5, levels=20)
    # plt.colorbar(label='Elevation')
    # nx.draw(G, pos, with_labels=False, node_size=500, node_color='lightblue', font_size=10)
    # # Add elevation map as background
    # plt.title('Branched Network')
    # plt.show()

    visualise_network(G, elevation_map, results=None, title='Initial Branched Network')

    # Check hydraulic feasibility


