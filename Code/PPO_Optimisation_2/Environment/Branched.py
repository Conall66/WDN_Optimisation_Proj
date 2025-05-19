
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

# Function to generate initital branched network, taking as inputs the size of the network, number of reservoirs, tanks and pumps, available pipe diamaters and roughness coefficients, and the elevation map

def generate_branched_network(
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
        # elevation_map,
        pipes,
        elevation_range = [0, 100],
        environment_type = 'urban',
        landscape_type = 'flat'):
    """
    Generate a branched water distribution network with specified parameters using random lobster. Assuming start in winter as Q1.

    Parameters:
    area_size (int): Size of the area for the network.
    num_nodes (int): Number of nodes in the network.
    num_reservoirs (int): Number of reservoirs in the network.
    num_tanks (int): Number of tanks in the network.
    initial_level (float): Initial water level in the tanks.
    min_level (float): Minimum water level in the tanks.
    max_level (float): Maximum water level in the tanks.
    tank_diameter (float): Diameter of the tanks.
    num_pumps (int): Number of pumps in the network.
    pump_type (str): Type of pump to be used.
    pump_power (float): Power of the pumps.
    roughness_coefficients (list): List of roughness coefficients for the pipes.
    base_demand (float): Base demand for the network.
    elevation_map (numpy.ndarray): Elevation map for the network.
    pipes (dict): Dictionary of available pipe diameters and their properties.
    elevation_range (list): Range of elevations for the network.
    environment_type (str): Type of environment ('urban', 'rural', etc.).

    Returns:
    graph (networkx.Graph): Generated branched water distribution network.

    """

    # Generate a random lobster network
    graph = nx.random_lobster(num_nodes, 0.5, 0.2) # In this function, the number of nodes determines how many junctions are along the backbone rather than in the network as a total

     # Generate layout for the graph using spring layout.
    # spring_layout by default positions nodes in a [0,1] x [0,1] area.
    # Using a seed ensures reproducible layouts.
    pos = nx.spring_layout(graph, seed=1) 

    # Scale coordinates to fit within the specified area_size.
    # This assumes area_size defines a square region [0, area_size] x [0, area_size].
    for node_key in graph.nodes(): # Iterate over all actual nodes in the graph
        # Get the [0,1] x [0,1] position from the layout for the current node
        x_layout, y_layout = pos[node_key]
        
        # Scale these positions to the dimensions of area_size
        scaled_x = x_layout * area_size[0]
        scaled_y = y_layout * area_size[1]
        
        # Assign the scaled coordinates to the node
        graph.nodes[node_key]['coordinates'] = (scaled_x, scaled_y)

    # Assign the largest diameter to the backbone, and randomly assign diameters to the other pipes
    # Find the pipe with the largest diameter
    largest_pipe_name = max(pipes, key=lambda p: pipes[p]['diameter'])
    largest_pipe_attrs = pipes[largest_pipe_name]

    pipe_names = list(pipes.keys()) # For random selection

    # Assign pipe properties to edges
    edge_id_counter = 0
    backbone_edges = []
    for u, v, data in graph.edges(data=True):
        data['id'] = f"{u}_{v}"
        # calculate length based on coordinates
        length = np.linalg.norm(np.array(graph.nodes[u]['coordinates']) - np.array(graph.nodes[v]['coordinates']))
        data['length'] = length
        edge_id_counter += 1
        # Check if both nodes are part of the original backbone (nodes 0 to num_nodes-1)
        # This assumes num_nodes in random_lobster refers to the backbone node count.
        # If num_nodes is the *total* number of nodes, this logic needs adjustment.
        # For random_lobster(n, p, k), n is the backbone length.
        if u < num_nodes and v < num_nodes: # Backbone edge
            data['diameter'] = largest_pipe_attrs['diameter']
            data['cost'] = largest_pipe_attrs['unit_cost'] * length
            data['carbon_emissions'] = largest_pipe_attrs['carbon_emissions']
            data['roughness'] = random.choice(roughness_coefficients)
            backbone_edges.append((u, v))
        else: # Other edges (legs)
            random_pipe_name = random.choice(pipe_names)
            random_pipe_attrs = pipes[random_pipe_name]
            data['diameter'] = random_pipe_attrs['diameter']
            data['cost'] = random_pipe_attrs['unit_cost'] * length
            data['carbon_emissions'] = random_pipe_attrs['carbon_emissions']
            data['roughness'] = random.choice(roughness_coefficients)

    # Add all nodes generated as junctions, with the demands split according to the environment type

    # If environment type is urban, 80% nodes are commercial and 20% are residential
    if environment_type == 'urban':
        num_residential = int(num_nodes * 0.2)
        num_commercial = int(num_nodes * 0.8)
    elif environment_type == 'rural':
        num_residential = int(num_nodes * 0.8)
        num_commercial = int(num_nodes * 0.2)

    # Shuffle the nodes to assign types
    node_ids = list(graph.nodes())
    random.shuffle(node_ids)

    assigned_res = 0
    assigned_com = 0
    for i, node_id in enumerate(node_ids):
        if assigned_res < num_residential:
            graph.nodes[node_id]['type'] = 'residential'
            assigned_res +=1
        else:
            graph.nodes[node_id]['type'] = 'commercial'
            assigned_com +=1


    # Generate the demand values and divide across the nodes
    _, populations = generate_sample_growth(start_year = 2025, end_year = 2050, start_population = 20000, end_population = 50000, num_samples = 100)
    # Generate demand profiles for residential and commercial use to describe the demand throughout the day
    residential_demand, commerical_demand = generate_demand_profiles(populations, uncertainty = None)
    # Isolate just initial year and quarter information
    residential_demand = residential_demand['Y2025Q1']
    commerical_demand = commerical_demand['Y2025Q1']

    print (f"Max esidential demand (L/s): {max(residential_demand['Demand'])}")
    print (f"Max commercial demand (L/s): {max(commerical_demand['Demand'])}")

    # Generate an elevation map
    elevation_map, peaks = generate_elevation_map(area_size=area_size, elevation_range=elevation_range, num_nodes=num_nodes, num_peaks=10, landscape_type=landscape_type)

    # For nodes in the graph, assign node type, an ID, elevation, base demand, minimum pressure and coordiates
    # Generate elevation map and peaks
    elevation_map, peaks, grid_size = generate_elevation_map(graph, elevation_range=elevation_range, num_nodes=len(graph.nodes()), num_peaks=10, landscape_type='hilly')

    # For nodes in the graph, assign node type, an ID, elevation, base demand, minimum pressure and coordinates
    for i in graph.nodes():
        graph.nodes[i]['id'] = f"J{i}"
        
        # Get coordinates from the stored 'coordinates' attribute
        x, y = graph.nodes[i]['coordinates']
        
        # Convert coordinates to proper grid indices (improved mapping)
        # Find the relative position in the area and map to grid indices
        grid_x = int((x / area_size[0]) * (grid_size))
        grid_y = int((y / area_size[1]) * (grid_size))
        
        # Ensure indices are within bounds
        grid_x = max(0, min(grid_x, grid_size-1))
        grid_y = max(0, min(grid_y, grid_size-1))
        
        # Assign elevation from the map
        graph.nodes[i]['elevation'] = elevation_map[grid_x, grid_y]
        graph.nodes[i]['min_pressure'] = 20 # Minimum pressure in the system

        if graph.nodes[i]['type'] == 'residential':
            graph.nodes[i]['demand'] = max(residential_demand['Demand']) * (1 + residential_demand['Uncertainty']) / num_residential # Multiply by uncertainty factor
        elif graph.nodes[i]['type'] == 'commercial':
            graph.nodes[i]['demand'] = max(commerical_demand['Demand']) * (1 + commerical_demand['Uncertainty'])/ num_commercial

        """Modelling demand as the highest daily demand in the year allows for a single step simulation to accelerate the training process. This is a simplification of the demand profile, but allows for a more efficient simulation. The demand profiles can be used to generate a more complex demand profile if needed."""

        # if graph.nodes[i]['type'] == 'residential':
        #     graph.nodes[i]['demand'] = base_demand * residential_demand['Residential Demand'] * residential_demand['Uncertainty'] # Multiply by uncertainty factor
        # elif graph.nodes[i]['type'] == 'commercial':
        #     graph.nodes[i]['demand'] = base_demand * commerical_demand['Commercial Demand'] * commerical_demand['Uncertainty']

    # For each reservoir in num_reservoirs, add to a peak in the elevation map
    # for i in range(num_reservoirs):
    #     # If there are no more peaks, add to a random point in the elevation map
    #     if peaks:
    #         peak = peaks.pop(0)
    #         x, y = peak
    #         # Convert to grid indices for elevation lookup
    #         grid_x = int(x*grid_size/area_size[0])
    #         grid_y = int(y*grid_size/area_size[1])
    #         graph.add_node(f"R{i}", id=f"R{i}", elevation=elevation_map[grid_x, grid_y], 
    #                      base_head=elevation_map[grid_x, grid_y], coordinates=(x, y))
    #     else:
    #         # Generate random coordinates within the area size
    #         random_x = random.randint(0, area_size[0])
    #         random_y = random.randint(0, area_size[1])
    #         # Convert to grid indices for elevation lookup
    #         grid_x = int(random_x*grid_size/area_size[0])
    #         grid_y = int(random_y*grid_size/area_size[1])
    #         graph.add_node(f"R{i}", id=f"R{i}", elevation=elevation_map[grid_x, grid_y], 
    #                      base_head=elevation_map[grid_x, grid_y], coordinates=(random_x, random_y))

    # # Connect each reservoir to its nearest node
    # for i in range(num_reservoirs):
    #     reservoir = f"R{i}"
    #     nearest_node = min(graph.nodes(), key=lambda n: np.linalg.norm(np.array(graph.nodes[n]['coordinates']) - np.array(graph.nodes[reservoir]['coordinates'])))
    #     # Calculate the length of the pipe based on the coordinates
    #     length = np.linalg.norm(np.array(graph.nodes[reservoir]['coordinates']) - np.array(graph.nodes[nearest_node]['coordinates']))
    #     graph.add_edge(
    #         reservoir, 
    #         nearest_node, 
    #         diameter=largest_pipe_attrs['diameter'], 
    #         cost=largest_pipe_attrs['unit_cost'] * length, 
    #         carbon_emissions=largest_pipe_attrs['carbon_emissions'], 
    #         roughness=random.choice(roughness_coefficients))

    # # For each tank in num_tanks, add tank to random peak in the elevation map
    # for i in range(num_tanks):
    #     # If there are no more peaks, add to a random point in the elevation map
    #     if len(peaks) > 0:
    #         peak = peaks.pop(0)
    #         graph.add_node(f"T{i}", id=f"T{i}", elevation=elevation_map[random_pos], base_head = elevation_map[random_pos], initial_level = initial_level, min_level = min_level, max_level = max_level, tank_diameter = tank_diameter, coordinates=(peak[0], peak[1]))
    #     else:
    #         random_pos = (random.randint(0, area_size[0]), random.randint(0, area_size[1]))
    #         graph.add_node(f"T{i}", id=f"T{i}", elevation=elevation_map[random_pos], base_head = elevation_map[random_pos], initial_level = initial_level, min_level = min_level, max_level = max_level, tank_diameter = tank_diameter, coordinates=(random_pos[0], random_pos[1]))

    # # Connect each tank to its nearest node
    # for i in range(num_tanks):
    #     tank = f"T{i}"
    #     nearest_node = min(graph.nodes(), key=lambda n: np.linalg.norm(np.array(graph.nodes[n]['coordinates']) - np.array(graph.nodes[tank]['coordinates'])))
    #     # Calculate the length of the pipe based on the coordinates
    #     length = np.linalg.norm(np.array(graph.nodes[tank]['coordinates']) - np.array(graph.nodes[nearest_node]['coordinates']))
    #     graph.add_edge(
    #         tank, 
    #         nearest_node, 
    #         diameter=largest_pipe_attrs['diameter'], 
    #         cost=largest_pipe_attrs['unit_cost'] * length, 
    #         carbon_emissions=largest_pipe_attrs['carbon_emissions'], 
    #         roughness=random.choice(roughness_coefficients))
        
    # # For each pump in num_pumps, add to random selection from backbone edges
    # for i in range(num_pumps):
    #     # If there are no more backbone edges, add to a random point in the elevation map
    #     if len(backbone_edges) > 0:
    #         # Replace one of the edges with a pumped edge of max diameter
    #         # Randomly sleect edge from the backbone edges
    #         edge = random.choice(backbone_edges)
    #         # Remove the edge from the list of backbone edges
    #         backbone_edges.remove(edge)
    #         # Add the pump to the edge
    #         graph.remove_edge(graph.edges(edge))
    #         graph.add_edge(
    #             edge[0], 
    #             edge[1], 
    #             pump_type=pump_type, 
    #             pump_power=pump_power, 
    #             diameter=largest_pipe_attrs['diameter'], 
    #             cost=largest_pipe_attrs['unit_cost'] * length, 
    #             carbon_emissions=largest_pipe_attrs['carbon_emissions'], 
    #             roughness=random.choice(roughness_coefficients))
            
    return graph, elevation_map

# Test function to generate a branched network
if __name__ == "__main__":
    # Define parameters for the network
    area_size = [100, 100]
    num_nodes = 20
    num_reservoirs = 2
    num_tanks = 2
    initial_level = 10
    min_level = 5
    max_level = 15
    tank_diameter = 1.0
    num_pumps = 2
    pump_type = 'Centrifugal'
    pump_power = 10.0
    roughness_coefficients = [0.01, 0.02, 0.03]
    base_demand = 1000
    # elevation_map = np.zeros((area_size, area_size))
    pipes = {
        'Pipe1': {'diameter': 0.1, 'unit_cost': 10, 'carbon_emissions': 1},
        'Pipe2': {'diameter': 0.2, 'unit_cost': 20, 'carbon_emissions': 2},
        'Pipe3': {'diameter': 0.3, 'unit_cost': 30, 'carbon_emissions': 3}
    }

    # Generate the network
    network, elevation_map = generate_branched_network(
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
        # elevation_map,
        pipes)

    # Plot the network
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(network, 'coordinates')
    
    # Create proper axis extents matching the network coordinates
    x_min = 0
    x_max = area_size[0]
    y_min = 0
    y_max = area_size[1]
    
    # Create meshgrid for the elevation map that matches the network area
     # Create properly scaled meshgrid for the elevation map
    x = np.linspace(0, area_size[0], elevation_map.shape[0])
    y = np.linspace(0, area_size[1], elevation_map.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Plot the elevation map with proper extents and transpose the elevation map for correct orientation
    plt.contourf(X, Y, elevation_map.T, cmap='terrain', alpha=0.5, levels=20, extent=[0, area_size[0], 0, area_size[1]])
    
    # Plot the network on top
    node_elevations = [network.nodes[n]['elevation'] for n in network.nodes()]
    nx.draw(network, pos, with_labels=True, node_size=50, 
            node_color=node_elevations, cmap='terrain', edge_color='gray')
    
    # Add these finishing commands to display the plot properly
    plt.colorbar(label='Elevation (m)')
    plt.title('Water Distribution Network with Elevation')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()  # This is essential to display the plot






