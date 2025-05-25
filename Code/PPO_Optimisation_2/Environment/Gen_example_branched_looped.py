
"""

In this file, we generate and store 2D examples of a minimum spanning tree branched network, the Hanoi network and the anytown network, plotting using the native wntr plotting function.

"""

import wntr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import networkx as nx

from Visualise_network import visualise_network
from Test_file import generate_test_lobster_wntr

def plot_network(inp_file, save_path, save_name):
    """
    Plot the network using wntr and save the figure.
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    # plt.figure()
    wntr.graphics.plot_network(wn, node_size=0.1)
    plt.savefig(os.path.join(save_path, save_name))
    # plt.close()

if __name__ == "__main__":
    # Define the input file and save path
    
    looped_networks = ['hanoi-3.inp', 'anytown-3.inp']
    branched_network, results = generate_test_lobster_wntr()

    for file in looped_networks:
        inp_file = os.path.join('Environment', 'Initial_networks', 'exeter', file)
        save_path = os.path.join('Environment', 'Example_branched_looped')
        save_name = f"{file.split('.')[0]}_network.png"
        
        # Plot the network
        # plot_network(inp_file, save_path, save_name)
        wn = wntr.network.WaterNetworkModel(inp_file)
        # Run simple hydraulic simulation
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        # Plot the network with results
        visualise_network(wn = wn, results = None, title = f"{file.split('.')[0]} Network", save_path = os.path.join(save_path, save_name), mode='2d')

    # Plot the branched network
    # plt.figure()
    # wntr.graphics.plot_network(branched_network, node_size=0.1)
    save_path = os.path.join('Environment', 'Example_branched_looped')
    save_name = "branched_network.png"
    visualise_network(wn = branched_network, results = None, title = "Branched Network", save_path = os.path.join(save_path, save_name), mode='2d')
    # plt.savefig(os.path.join(save_path, save_name))
    # plt.close()