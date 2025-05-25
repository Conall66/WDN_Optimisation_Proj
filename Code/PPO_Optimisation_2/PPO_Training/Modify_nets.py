
"""

In this file, we take the initial networks and convert units to match. we then assign pipe diameters closest to those of the originial network from the discrete set

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import random
import wntr
from wntr.network.io import write_inpfile
import time
from wntr.graphics.network import plot_network

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# Function to convert units if they are not metric
def convert_units(inp_file, out_file):
    """
    Function takes .inp file as input, chceks units and convert them to metric if necessary
    
    Parameters:
    inp_file (str): Path to the input .inp file
    out_file (str): Path to the output .inp file

    Returns:
    None
    
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    # Check if the units are not metric
    units = wn.options.hydraulic.inpfile_units

    print(f"Current units in the network: {units}")

    if units != 'CMH':
        # Convert units to metric
        wn.options.hydraulic.inpfile_units = 'CMH'
        # wn.options.hydraulic.inpfile_pressure_units = 'M'
        units = 'CMH'
        
    # Save the modified network to a new file
    write_inpfile(wn, out_file)

    return wn, units

# Function to map pipe diameters to closest in the discrete set

def map_pipe_diameters(inp_file, discrete_diameters):
    """
    Function to map pipe diameters in the inp_file to the closest diameter in the discrete_diameters set.
    
    Parameters:
    inp_file (str): Path to the input .inp file
    discrete_diameters (list): List of discrete pipe diameters to map to

    Returns:
    None
    
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    pipes = wn.pipes
    
    for pipe_id, pipe in wn.pipes():
        original_diameter = pipe.diameter
        closest_diameter = min(discrete_diameters, key=lambda x: abs(x - original_diameter))
        pipe.diameter = closest_diameter
    
    # Save the modified network
    write_inpfile(wn, inp_file)

def test_hydraulic_analysis(inp_file):
    """
    Function to run a hydraulic analysis on the inp_file and return the results.
    
    Parameters:
    inp_file (str): Path to the input .inp file

    Returns:
    results: Hydraulic analysis results
    
    """
    wn = wntr.network.WaterNetworkModel(inp_file)

    wn.options.time.duration = 24*3600  # Steady state simulation
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600

    # extract just file name from path
    file_name = os.path.basename(inp_file)

    try:
        start_time = time.time()
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Hydraulic simulation completed for network: {file_name}")
        print(f"Run time for hydraulic simulation: {run_time:.4f} seconds")
    except Exception as e:
        print(f"Error running hydraulic simulation: {e}")
        return None
    
    return results

# Example run

if __name__ == "__main__":
    
    # Define input file names
    files = ['anytown-3.inp', 'hanoi-3.inp']
    script = os.path.dirname(__file__)

    # Enable pumps with realistic curves for the Anytown network
    # anytown_file = os.path.join(script, 'Initial_networks', 'exeter', 'anytown-3.inp')
    # enable_pumps_with_curve(anytown_file)

    # First test hydraulic analysis on initial networks
    for file in files:
        file_path = os.path.join(script, 'Initial_networks', 'exeter', file)
        # Run hydraulic anlaysis
        try:
            wn = wntr.network.WaterNetworkModel(file_path)
            results = run_epanet_simulation(wn)
            performance_metrics = evaluate_network_performance(wn, results)
            # Extract energy consumption value from initial network
            energy_consumption = performance_metrics['total_energy_consumption']
            print(f"Initial energy consumption for {file}: {energy_consumption} kWh")
        except Exception as e:
            print(f"Error running hydraulic analysis for {file}: {e}")
            continue
    
    print("----------------------------------")

    # --------------------------------

    # Test on imported anytown network
    # new_file = os.path.join(script, 'Initial_networks', 'Anytown.inp')
    # try:
    #     wn = wntr.network.WaterNetworkModel(new_file)
    #     results = run_epanet_simulation(wn)
    #     performance_metrics = evaluate_network_performance(wn, results)
    #     # Extract energy consumption value from initial network
    #     energy_consumption = performance_metrics['total_energy_consumption']
    #     print(f"Initial energy consumption for Anytown.inp: {energy_consumption} kWh")
    # except Exception as e:
    #     print(f"Error running hydraulic analysis for {file}: {e}")

    # print("----------------------------------")

    for file in files:
        file_path = os.path.join(script, 'Initial_networks', 'exeter', file)
        save_path = os.path.join(script, 'Modified_nets', file)
    
        # Convert units
        wn, units = convert_units(file_path, save_path)

        # if 'anytown' in file.lower():
        #     print(f"Adding pump curves to {file}")
        #     wn = add_pump_curves_to_anytown(save_path)

        # Map pipe diameters to a discrete set
        discrete_diameters = [0.3048, 0.4064, 0.508, 0.609, 0.762, 1.016]
        map_pipe_diameters(save_path, discrete_diameters)

        # If anytown, add pump pattern to accumulate energy consumption

        # Test hydraulic analysis
        results = test_hydraulic_analysis(save_path)
        # Display the results in graph form (node pressure added to the network)
        
        # Print the units to verify
        print(f"Units of the network: {units}")
    
