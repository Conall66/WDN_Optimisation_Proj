
"""

In this file we check the functionality of various components of the pipe sizing optimisation project, including:

- Node connections (are all nodes connected, even if by a pipe with diameter 0?)
- Hydrualic simulation (all metrics are calculating and coming out at reasonable values)
- Environment (is the environment updating correctly with each time step?)
- PPO agent (is the reward increasing generally with each time step?)
- Reward hacking (is the agent taking repeated actions to gain a reward?)
- Illegal actions (is the agent taking illegal actions?)

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

from Hydraulic_Model import *
from Environment import *

# Test 1: Generate initial graph for all network types and calculate hydraulic metrics witout return an error

def test_model_generation():

    """
    Test the generation of the hydraulic model and calculation of metrics.
    """

    # Arbitrary set of inputs
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

    try:
        # Generate the graph
        graph = generate_initial_network(num_nodes, num_reservoirs, area_size, num_tanks, num_pumps, pipe_diameters, roughness_coeffs, min_elevations, max_elevations, min_demand, max_demand, min_pipe_length, max_pipe_length, min_level, max_level, diameter, pump_parameter, pump_type, network_type)
        # print("Graph generated successfully")
    except Exception as e:
        print(f"Error generating graph: {e}")
        return False

    try:
        # Convert the graph to a wntr model
        wn = convert_graph_to_wntr(graph)
        # print("Graph converted to wntr model successfully")
    except Exception as e:
        print(f"Error converting graph to wntr model: {e}")
        return False
    
    # Run hydrualic analysis
    try:
        results = run_epanet_simulation(wn, 3600, 60) # Shorter simulation to test
        # print("EPANET simulation ran successfully")

    except Exception as e:
        print(f"Error running epanet simulation on wn graph: {e}")
        return False

    # Extract metrics from epanet simulation
    try:
        metrics = evaluate_network_performance(wn, results, min_pressure=20, final_time=3600)
        # print("Hydraulic performance metrics extracted successfully")
    except Exception as e:
        print(f"Failure in extracting hydraulic performance metrics: {e}")
        return False
    
    return True


# Check all tanks are far from each other and reservoirs

# Test the hydraulic model results are feasible for a particular set of inputs

# Run the tests
if __name__ == "__main__":
    print(f"test_model_generated: {test_model_generation()}")