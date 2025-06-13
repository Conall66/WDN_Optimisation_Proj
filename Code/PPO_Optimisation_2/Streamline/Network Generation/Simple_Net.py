
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
# from Visualise import plot_network_3d

def generate_simple_network():

    # Create a new water network model
    wn = wntr.network.WaterNetworkModel()

    # Add reservoir in position (0, 0)
    wn.add_reservoir('R1', base_head=15, coordinates=(0, 0)) # Elevated above

    # Add 5 junctions and connect each to the last
    wn.add_junction('J1', base_demand=0.02, coordinates=(10, 0), elevation = 0)
    wn.add_junction('J2', base_demand=0.02, coordinates=(20, 10), elevation = 2)
    wn.add_junction('J3', base_demand=0.02, coordinates=(30, 10), elevation = 3)
    wn.add_junction('J4', base_demand=0.02, coordinates=(40, 20), elevation = 1)
    wn.add_junction('J5', base_demand=0.02, coordinates=(50, 20), elevation = 0)

    # Add pipes
    wn.add_pipe('P1', 'R1', 'J1', length=1000, diameter=0.300, roughness=100) # 1000m, 0.3m diameter, 100 roughness
    wn.add_pipe('P2', 'J1', 'J2', length=1000, diameter=0.300, roughness=60)
    wn.add_pipe('P3', 'J2', 'J3', length=1000, diameter=0.300, roughness=60)
    wn.add_pipe('P4', 'J3', 'J4', length=1000, diameter=0.300, roughness=60)
    wn.add_pipe('P5', 'J4', 'J5', length=1000, diameter=0.300, roughness=60)

    return wn

def generate_net_evolution(wn, last_junction):
    """
    From an initial network, it generates a new network with additional junctions (with a 10% probability) - any new pipes are connected to the last junction
    """

    new_wn = deepcopy(wn)  # Create a copy of the current network to modify
    
    # Increase demand at all existing junctions by 10% of the original value
    original_demand = 0.02
    demand_increase = 0.1 * original_demand  # 10% of original value
    
    # Update demands for all junctions
    for junction_name in new_wn.junction_name_list:
        junction = new_wn.get_node(junction_name)
        # Clear existing demands and add the updated demand
        junction.demand_timeseries_list.clear()
        current_base_demand = sum(d.base_value for d in junction.demand_timeseries_list) if junction.demand_timeseries_list else junction.demand_timeseries_list[0].base_value if junction.demand_timeseries_list else 0
        new_demand = current_base_demand + demand_increase
        junction.add_demand(new_demand, 'Base Demand')
    
    # Only create a copy if a new junction is being added
    if random.random() < 0.1:  # 10% chance to add a new junction
        
        # Define new junction and pipe names based on the current count
        new_junction_name = f'J{len(new_wn.junction_name_list) + 1}'
        new_pipe_name = f'P{len(new_wn.pipe_name_list) + 1}'
        
        # Get coordinates of the last junction to place the new one nearby
        last_node_coords = new_wn.get_node(last_junction).coordinates
        new_junction_coordinates = (last_node_coords[0] + random.randint(10, 20), last_node_coords[1] + random.randint(-10, 10))
        
        # Add the new junction and pipe
        new_wn.add_junction(new_junction_name, base_demand=0.02, coordinates=new_junction_coordinates)
        new_wn.add_pipe(new_pipe_name, last_junction, new_junction_name, length=1000, diameter=0.300, roughness=100)
        
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
        plot_network(wn, node_attribute=pressure, title=f'Simple Network at Step {step}', node_size=100, node_colorbar_label='Pressure (m)', node_labels = True, show_plot=False)
    else:
        pressure = np.zeros(len(wn.junction_name_list))  # Default to zero if no results

    plt.savefig(f'Plots/Simple_Nets/simple_network_{step}.png')

    plt.close()  # Close the plot to avoid displaying it immediately

    # plt.show()

def run_sim(wn):

    """
    Run the wntr hydraulic simulation and return the results
    """

    wn.options.time.duration = 24*3600
    wn.options.time.hydraulic_timestep = 3600  # 1 hour timestep
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.inpfile_units = 'CMH'  # Cubic meters per hour
    wn.options.hydraulic.accuracy = 0.01
    wn.options.hydraulic.trials = 100 # Attempts the simulation 100 times to ensure convergence
    wn.options.hydraulic.headloss = 'H-W'  # Hazen-Williams headloss
    wn.options.hydraulic.demand_model = 'DDA'  # Demand driven analysis
    wn.options.energy.global_efficiency = 100.0  # Global efficiency for energy calculations

    wn.options.energy.global_price = 0.26  # Global price for energy calculations

    # Run the hydraulic simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    return results

def test_pipe_diameters(network_file, min_diameter=0.1, max_diameter=1.0, step=0.05):
    """
    Test the effect of different pipe diameters on pressure deficit in a water network.
    
    Parameters:
    -----------
    network_file : str
        Path to the network .inp file to test
    min_diameter : float
        Minimum pipe diameter to test (in meters)
    max_diameter : float
        Maximum pipe diameter to test (in meters)
    step : float
        Step size for diameter increments
        
    Returns:
    --------
    DataFrame with diameter values and corresponding pressure metrics
    """
    # Load the network
    wn = wntr.network.WaterNetworkModel(network_file)
    
    # Create result storage
    results = []
    diameters = np.arange(min_diameter, max_diameter + step, step)
    
    # Test each diameter
    for diameter in diameters:
        # Set all pipe diameters to the current test value
        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            pipe.diameter = diameter
        
        try:
            # Run simulation
            sim_results = run_sim(wn)
            
            # Get pressure at final timestep for all nodes
            pressures = sim_results.node['pressure'].iloc[-1]
            
            # Calculate deficit (assuming 20m is the minimum desired pressure)
            min_desired_pressure = 20
            deficits = {node: max(0, min_desired_pressure - pressure) 
                       for node, pressure in pressures.items() 
                       if node in wn.junction_name_list}
            
            # Store metrics
            results.append({
                'diameter': diameter,
                'min_pressure': pressures[wn.junction_name_list].min(),
                'max_pressure': pressures[wn.junction_name_list].max(),
                'avg_pressure': pressures[wn.junction_name_list].mean(),
                'total_deficit': sum(deficits.values()),
                'max_deficit': max(deficits.values()) if deficits else 0,
                'nodes_with_deficit': sum(1 for d in deficits.values() if d > 0)
            })
            
            print(f"Tested diameter {diameter:.2f}m - Min pressure: {results[-1]['min_pressure']:.2f}m")
            
        except Exception as e:
            print(f"Simulation failed for diameter {diameter:.2f}m: {e}")
            # Add failed result
            results.append({
                'diameter': diameter,
                'min_pressure': None,
                'max_pressure': None,
                'avg_pressure': None,
                'total_deficit': None,
                'max_deficit': None,
                'nodes_with_deficit': None
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Pressure metrics
    plt.subplot(2, 1, 1)
    plt.plot(results_df['diameter'], results_df['min_pressure'], 'b-', label='Min Pressure')
    plt.plot(results_df['diameter'], results_df['avg_pressure'], 'g-', label='Avg Pressure')
    plt.plot(results_df['diameter'], results_df['max_pressure'], 'r-', label='Max Pressure')
    plt.axhline(y=min_desired_pressure, color='k', linestyle='--', label='Min Required')
    plt.xlabel('Pipe Diameter (m)')
    plt.ylabel('Pressure (m)')
    plt.title('Pressure vs Pipe Diameter')
    plt.legend()
    plt.grid(True)
    
    # Deficit metrics
    plt.subplot(2, 1, 2)
    plt.plot(results_df['diameter'], results_df['total_deficit'], 'b-', label='Total Deficit')
    plt.plot(results_df['diameter'], results_df['max_deficit'], 'r-', label='Max Deficit')
    plt.plot(results_df['diameter'], results_df['nodes_with_deficit'], 'g-', label='Nodes with Deficit')
    plt.xlabel('Pipe Diameter (m)')
    plt.ylabel('Pressure Deficit / Node Count')
    plt.title('Pressure Deficit vs Pipe Diameter')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Plots/diameter_pressure_test.png')
    plt.close()
    
    # Save results to CSV
    results_df.to_csv('Networks/Hydraulic_Results/diameter_pressure_test.csv', index=False)
    
    return results_df

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
    # plot_network_3d(wn=wn, step=0, results = results)

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
            # sim = wntr.sim.EpanetSimulator(wn)
            # results = sim.run_sim()

            results = run_sim(wn)

            # Store a dictionary of the results in the dataframe
            results_df.loc[step, 'results'] = results

            print(f"Step {step}: Hydraulic simulation successful.")
            write_inpfile(wn, f'Networks/Simple_Nets/simple_network_{step}.inp')
            plot_network_wn(wn, step, results)
            # plot_network_3d(wn=wn, step=step, results = results)

        except Exception as e:
            results_df.loc[step, 'status'] = f'failed: {str(e)}'
            print(f"Step {step}: Hydraulic simulation failed with error: {e}. Continuing to next step.")

    # Save the results dataframe to a CSV file
    # Save only the columns that can be easily serialised to CSV
    save_df = results_df[['n_junctions', 'n_pipes', 'status']].copy()
    save_df.to_csv('Networks/Hydraulic_Results/network_evolution_results.csv')

    # Print the hydraulic units
    # units = wn.options.hydraulic.inpfile_units
    # print(f"Hydraulic simulation units: {units}")
    
    # Return the full dataframe with hydraulic results for further analysis if needed
    return results_df

# Check hydraulic simulation

if __name__ == "__main__":

    results = main()

    print("Testing pressure deficits with different pipe diameters...")
    test_results = test_pipe_diameters(
        'Networks/Simple_Nets/simple_network_50.inp',  # Use the most complex network
        min_diameter=0.1,
        max_diameter=1.0,
        step=0.05
    )
    print("Test complete. Results saved to Networks/Hydraulic_Results/diameter_pressure_test.csv")