
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

# Test 1: Generate initial graph and calculate hydraulic metrics witout return an error

def test_model_generation():

    """
    Test the generation of the hydraulic model and calculation of metrics.
    """

    # Generate a random graph
    try:
        initial_network = generate_initial_network(
            network_size=10,
            topology=NetworkTopology.LOOPED, # Arbitrarily pick a topology
            available_diameters=[0.1, 0.2, 0.3, 0.4, 0.5], # Arbitrarily pick diameters
            available_roughness=[0.01, 0.02, 0.03, 0.04, 0.05], # Arbitrarily pick roughness values
            num_reservoirs=1,
            num_pumps=1,
            num_tanks=1,
        )
        # Visualise network
        initial_network.visualise("Initial_Network")

        # Display the nodes
        # print(f"Nodes in the network: {initial_network.graph.nodes(data=True)}")

    except Exception as e:
        print(f"Error generating initial network: {e}")
        return False
    
    # Print edge attributes
    # print(f"Edges attributes: {initial_network.graph.edges(data=True)}")

    # Convert the graph to an EPANET .inp file
    try:
        wn = convert_graph_to_wntr(initial_network.graph)
        # Run the EPANET simulation
    except Exception as e:
        print(f"Error converting graph to wntr file: {e}")
        return False

    try:
        results = run_epanet_simulation(wn, duration=3600, time_step=60) # Short simulation for testing
    except Exception as e:
        print(f"Error during EPANET simulation: {e}")
        return False
    
    # Check if results can be converted to evaluation metrics
    try:
        metrics = evaluate_network_performance(initial_network.graph, results)
    except Exception as e:
        print(f"Error calculating hydraulic metrics: {e}")
        return False
    
    print("Hydraulic metrics calculated successfully.")
    print("Metrics:", metrics)
    return True

# Run the tests
if __name__ == "__main__":
    test_model_generation()