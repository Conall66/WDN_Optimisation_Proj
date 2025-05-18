
"""

In this file, we collate all the components for the pipe sizing optimisation project, including:
- Generating the initial network, final network and all intermediate networks
- Generating hydraulic results from a given network with pipe features
- Training the PPO agent to optimise the configuration of the network
- Visualising the network and the results
- Evaluating the performance of the network

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
import time
import copy

from Hydraulic_Model import *
from Environment import generate_initial_network
from Visualise_network import *

# Function to generate the initial networks and all intermediate networks

def generate_networks(num_nodes, num_reservoirs, area_size, num_tanks, num_pumps, pipe_diameters, roughness_coeffs, min_elevations, max_elevations, min_demand, max_demand, min_pipe_length, max_pipe_length, min_level, max_level, diameter, pump_parameter, pump_type, network_type):
    """
    Generate the initial network and all intermediate networks.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes in the network
    num_reservoirs : int
        Number of reservoirs in the network
    area_size : int
        Size of the area in which the network is located
    num_tanks : int
        Number of tanks in the network
    num_pumps : int
        Number of pumps in the network
    pipe_diameters : list
        List of pipe diameters to be used in the network
    roughness_coeffs : list
        List of roughness coefficients to be used in the network
    min_elevations : float
        Minimum elevation for nodes in the network
    max_elevations : float
        Maximum elevation for nodes in the network
    min_demand : float
        Minimum demand for nodes in the network
    max_demand : float
        Maximum demand for nodes in the network
    min_pipe_length : float
        Minimum length of pipes in the network
    max_pipe_length : float
        Maximum length of pipes in the network
    min_level : float
        Minimum level for tanks in the network
    max_level : float
        Maximum level for tanks in the network
    
    Returns:
    ---------
    graph : NetworkX graph object
        The generated water distribution network as a graph object.
    
    """

    # Generate the initial network
    graph = generate_initial_network(num_nodes, num_reservoirs, area_size, num_tanks, num_pumps, pipe_diameters, roughness_coeffs, min_elevations, max_elevations, min_demand, max_demand, min_pipe_length, max_pipe_length, min_level, max_level, diameter, pump_parameter, pump_type, network_type)
    
    # Visualise the initial network
    visualise_network(graph, title="Initial Network")

    # Run hydraulic simulation on the initial network
    water_network = convert_graph_to_wntr(graph)
    results = run_epanet_simulation(water_network, (3600*24), 3600)
    metrics = evaluate_network_performance(water_network, results)

    print(f"Initial Network Performance Metrics: {metrics}")

    # Visualise the initial network with node pressures and head loss values
    visualise_network(graph, results=results, title="Initial Network with Hydraulic Results")
    
    return graph

if __name__ == "__main__":
    
    # Generate the initial network
    graph = generate_networks(num_nodes=20, num_reservoirs=1, area_size=1000, num_tanks=1, num_pumps=1, pipe_diameters=[100.0, 150.0, 200.0], roughness_coeffs=[0.01, 0.02, 0.03], min_elevations=10, max_elevations=50, min_demand=5, max_demand=20, min_pipe_length=25, max_pipe_length=200, min_level=5, max_level=15, diameter=0.1, pump_parameter=100, pump_type='POWER', network_type='looped')