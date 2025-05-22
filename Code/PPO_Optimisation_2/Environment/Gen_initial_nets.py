
"""

In this file, we generate the initial networks, run hydraulic analysis and store the network results and visualisations.

"""

# Import neccesary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import wntr

from Visualise_network import visualise_network
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# For each file in the networks folder, create a subfolder with the same name

# Tets import a single file and visualise
# file_1 = os.path.join('Newtorks', 'exeter', 'sampletown.inp.inp')
# Check file exist in directory

def network_analysis(file_path, save_directory, file_name):
    """
    Function to perform network analysis on a given file.
    
    Parameters:
    - file_path: str, path to the directory containing the input file
    - save_directory: str, path to the directory where results will be saved
    - file_name: str, name of the input file (without extension)
    
    Returns:
    - wn: WaterNetworkModel object
    - results: hydraulic results from the network
    """
    
    # Load the network
    wn = wntr.network.WaterNetworkModel(file_path)
    
    # Get hydraulic results from the network
    results = run_epanet_simulation(wn)
    
    # Get hydraulic performance
    metrics = evaluate_network_performance(wn, results)

    # Save a network visualisation to the specified directory with file name
    save_path = os.path.join(save_directory, f"{file_name}.png")
    visualise_network(wn, results, title=f"Water Distribution Network {file_name}", save_path=save_path, mode='3d')
    
    return wn, results, metrics

if __name__ == "__main__":
    # Set the directory where the networks are stored
    # Get the current script directory

    script = os.path.dirname(__file__)
    # Get all networks in the networks folder
    networks_dir = os.path.join(script, 'Initial_networks')
    network_files = [f for f in os.listdir(networks_dir) if f.endswith('.inp')]

    # Process all .inp files in all subdirectories of Networks
    processed_files = 0
    error_files = 0

    save_directory = os.path.join(script, 'Initial_network_visualisation')

    for root, dirs, files in os.walk(networks_dir):
        for file in files:
            if file.endswith('.inp'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0] # Get file name without extension
                success = network_analysis(file_path, save_directory, file_name)
                if success:
                    processed_files += 1
                else:
                    error_files += 1

    print(f"Processing complete. Successfully processed {processed_files} files. Errors: {error_files}")

