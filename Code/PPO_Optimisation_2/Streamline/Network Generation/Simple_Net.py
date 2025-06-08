
"""

In this script, we create a simple 5 junction network with 1 reservoir, and the 50 growth network states to test the performance of the DRL agent training.

"""

import wntr
from wntr.graphics.network import plot_network
from wntr.network.io import write_inpfile

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import random
from copy import deepcopy
import os
import sys

script = os.path.dirname(__file__)
parents_dir = os.path.dirname(script)
sys.path.append(parents_dir)

from Hydraulic import run_epanet_sim

def generate_simple_network():

    # Create a new water network model
    wn = wntr.network.WaterNetworkModel()

    # Add reservoir in position (0, 0)
    wn.add_reservoir('R1', base_head=100, coordinates=(0, 0)) # Elevated above

    # Add 5 junctions and connect each to the last
    wn.add_junction('J1', base_demand=1, coordinates=(10, 0))
    wn.add_junction('J2', base_demand=1, coordinates=(20, 10))
    wn.add_junction('J3', base_demand=1, coordinates=(30, 10))
    wn.add_junction('J4', base_demand=1, coordinates=(40, 20))
    wn.add_junction('J5', base_demand=1, coordinates=(50, 20))

    # Add pipes
    wn.add_pipe('P1', 'R1', 'J1', length=100, diameter=0.3, roughness=100) # 100m, 0.3m diameter, 100 roughness
    wn.add_pipe('P2', 'J1', 'J2', length=100, diameter=0.3, roughness=100)
    wn.add_pipe('P3', 'J2', 'J3', length=100, diameter=0.3, roughness=100)
    wn.add_pipe('P4', 'J3', 'J4', length=100, diameter=0.3, roughness=100)
    wn.add_pipe('P5', 'J4', 'J5', length=100, diameter=0.3, roughness=100)

    return wn

def generate_net_evolution(wn, last_junction):
    """
    From an initial network, it generates a new network with additional junctions (with a 10% probability) - any new pipes are connected to the last junction
    """

    new_wn = wntr.network.WaterNetworkModel()

    # Only create a copy if a new junction is being added
    if random.random() < 0.1:  # 10% chance to add a new junction
        
        # Create a deep copy of the network before modifying it
        new_wn = deepcopy(wn)
        
        # Define new junction and pipe names based on the current count
        new_junction_name = f'J{len(new_wn.junction_name_list) + 1}'
        new_pipe_name = f'P{len(new_wn.pipe_name_list) + 1}'
        
        # Get coordinates of the last junction to place the new one nearby
        last_node_coords = new_wn.get_node(last_junction).coordinates
        new_junction_coordinates = (last_node_coords[0] + random.randint(10, 20), last_node_coords[1] + random.randint(-10, 10))
        
        # Add the new junction and pipe
        new_wn.add_junction(new_junction_name, base_demand=10, coordinates=new_junction_coordinates)
        new_wn.add_pipe(new_pipe_name, last_junction, new_junction_name, length=100, diameter=0.3, roughness=100)
        
        # Update the last junction to the newly created one
        last_junction = new_junction_name

        return new_wn, last_junction
    
    # If no junction is added, return the original network and last junction
    return wn, last_junction

def plot_network_wn(wn, step, results = None):
    """
    Plot the water network using WNTR's built-in plotting function and save to the Networks folder.
    """

    # Extract pressure values from results
    if results is not None:
        pressure = dict(zip(wn.node_name_list, results.node['pressure'].loc[results.node['pressure'].index[-1], :]))
        plot_network(wn, node_attribute=pressure, title=f'Simple Network at Step {step}', node_size=100, node_colorbar_label='Pressure (m)', show_plot=False)
    else:
        pressure = np.zeros(len(wn.junction_name_list))  # Default to zero if no results

    plt.savefig(f'Plots/Simple_Nets/simple_network_{step}.png')

    plt.close()  # Close the plot to avoid displaying it immediately

    # plt.show()

def main():
    # Create DataFrame to store network information and hydraulic results
    results_df = pd.DataFrame(
        index=range(51),  # 0-50 steps
        columns=['n_junctions', 'n_pipes', 'results', 'status']
    )

    # Generate the initial simple network
    wn = generate_simple_network()

    # Save each network to the Networks/Simple_Nets folder
    write_inpfile(wn, 'Networks/Simple_Nets/simple_network_0.inp')

    # Record initial network stats
    results_df.loc[0, 'n_junctions'] = len(wn.junction_name_list)
    results_df.loc[0, 'n_pipes'] = len(wn.pipe_name_list)
    results_df.loc[0, 'status'] = 'success'

    # Try running hydraulic simulation for initial network
    try:
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        results_df.loc[0, 'results'] = results
        print(f"Step 0: Hydraulic simulation successful.")
    except Exception as e:
        results_df.loc[0, 'status'] = f'failed: {str(e)}'
        print(f"Step 0: Hydraulic simulation failed with error: {e}.")

    # Plot the initial network
    plot_network_wn(wn=wn, step=0, results = results)

    # Generate and plot 50 network states
    last_junction = 'J5'
    for step in range(1, 51):
        wn, last_junction = generate_net_evolution(wn, last_junction)
        
        # Record network stats
        results_df.loc[step, 'n_junctions'] = len(wn.junction_name_list)
        results_df.loc[step, 'n_pipes'] = len(wn.pipe_name_list)
        results_df.loc[step, 'status'] = 'success'

        # Try running hydraulic simulation, if fails continue to next step
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            # Store a dictionary of the results in the dataframe
            results_df.loc[step, 'results'] = results

            print(f"Step {step}: Hydraulic simulation successful.")
            write_inpfile(wn, f'Networks/Simple_Nets/simple_network_{step}.inp')
            plot_network_wn(wn, step, results)

        except Exception as e:
            results_df.loc[step, 'status'] = f'failed: {str(e)}'
            print(f"Step {step}: Hydraulic simulation failed with error: {e}. Continuing to next step.")

    # Save the results dataframe to a CSV file
    # Save only the columns that can be easily serialised to CSV
    save_df = results_df[['n_junctions', 'n_pipes', 'status']].copy()
    save_df.to_csv('Networks/Hydraulic_Results/network_evolution_results.csv')
    
    # Return the full dataframe with hydraulic results for further analysis if needed
    return results_df

# Check hydraulic simulation

if __name__ == "__main__":

    results = main()