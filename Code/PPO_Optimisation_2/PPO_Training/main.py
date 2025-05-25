
"""

This is the main file, in which initial networks ar extracted, and demand values updated according to forecasts. The PPO agent then takes as input these networks and learns to allocate pipe diameters to the network over time. 

"""
# Import necessary libraries
import os
import wntr
import random
from wntr.network.io import write_inpfile
import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
import shutil
from wntr.graphics.network import plot_network

from Visualise_network import visualise_network
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# Extract initial networks and run initial hydraulic analysis to set sa baseline for agent performance
# Modified network exist and store in modified_nets folder
script = os.path.dirname(__file__)
initial_networks_folder = os.path.join(script, 'Modified_nets')
# Files are in the initial networks folder

# Visualise the initial networks with node pressure coloured
for file in os.listdir(initial_networks_folder):
    # Perform hydraulic analysis
    # Extract file_name from folder
    if file.endswith('.inp'):
        # file_name = os.path.splitext(file)[0]  # Get the file name without extension
        inp_file = os.path.join(initial_networks_folder, file)
        print(f"Processing file: {file}")
        wn = wntr.network.WaterNetworkModel(inp_file)
        print(f"Processing network: {file}")

        results = run_epanet_simulation(wn)
        # sim = wntr.sim.EpanetSimulator(wn)
        # results = sim.run_sim()
        # print(f"Hydraulic analysis completed for network: {file}")
        # Initial performance metrics
        performance_metrics = evaluate_network_performance(wn, results)
        # print energy consumption values
        # Extract final key from dictionary of perforamcen metrics
        # print(f"Performance metrics for {file}: {performance_metrics}")
        print(f"Final energy consumption: {performance_metrics['total_energy_consumption']} kWh")
        # Visualise the network
        plots_path = os.path.join(script, 'Plots', 'Initial_' + file.replace('.inp', '.png'))
        visualise_network(wn = wn, results = results, title = f"Initial Network: {file}", save_path = plots_path, mode = '2d', show = False)

        print("----------------------------------")

        """
        Noticeably, the hanoi network is completely gravity fed and so always has an energy consumption value of 0kWh in theory.
        """
    
# Generate scenarios of each water network

# Create set of networks and store in separate folders to be accessed at by random by episode

# For each episode, select a random network from the set and run the agent to refine policy

# Extract agent performance by reward, run time and number of iterations

# Tabulate the performance of the agent by network topology, demand scenario, etc.

# Extract how pipe bursts/age affected agent performance

# Import a GA agent to compare performance with the PPO agent

# Extract a larger network from the initial networks folder and test both agents extracting performance metrics again