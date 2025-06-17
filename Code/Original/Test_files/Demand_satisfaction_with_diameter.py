"""
Fix the pipe diameters and plot how demand satisfaction changes 
for the basic hanoi and anytown networks across the time steps.
"""

import os
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
import sys
from copy import deepcopy # Import deepcopy

# Add parent directory to path so we can import from parent modules
# Ensure this path is correct relative to your script's location
# Assuming this script is in a subdirectory of the project root,
# and Hydraulic_Model.py is in the project root.
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
except ImportError:
    print("Error: Could not import Hydraulic_Model. Ensure the script is in the correct directory structure.")
    sys.exit(1)


def set_pipe_diameters_for_analysis(wn_original, diameter_value, network_name_identifier):
    """Sets relevant pipes in the network to the specified diameter."""
    wn = deepcopy(wn_original) # Work on a copy
    
    exclude_pipes = []
    if 'hanoi' in network_name_identifier.lower():
        exclude_pipes = ['1', '2'] # Example: Main source pipes for Hanoi
    elif 'anytown' in network_name_identifier.lower():
        exclude_pipes = ['4'] # Example: Main source pipe for Anytown
        # For Anytown, pump '4' is often a critical element, but pumps don't have 'diameter'.
        # The pipe connected to a reservoir or pump might be what's intended.
        # Let's assume '4' refers to a pipe name to exclude from diameter changes.
        # If it's a pump, this check won't affect it as pumps are not in wn.pipes().

    for pipe_name, pipe_object in wn.pipes():
        if pipe_name not in exclude_pipes:
            pipe_object.diameter = diameter_value
    return wn

def plot_ds_with_t_corrected(scenario_folder_path, pipe_diameters_to_test, scenario_name):
    """
    Processes a scenario folder: for each .inp file (time step),
    it tests different fixed pipe diameters, runs simulations, 
    and plots demand satisfaction over time with total demand on a secondary y-axis.
    """
    
    # Store results: {diameter_value: [ds_ratio_step0, ds_ratio_step1, ...]}
    results_by_diameter = {diam: [] for diam in pipe_diameters_to_test}
    time_step_labels = [] # Store file names or derived step numbers for x-axis
    total_demands = []  # Store total demand at each time step
    
    # Get sorted list of .inp files representing time steps
    inp_files = [f for f in os.listdir(scenario_folder_path) if f.endswith('.inp') and f.startswith('Step_')]
    inp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by step number
    print(f"Sorted .inp files: {inp_files}")

    if not inp_files:
        print(f"No 'Step_*.inp' files found in {scenario_folder_path}")
        return

    print(f"\nProcessing scenario: {scenario_name} in folder: {scenario_folder_path}")

    for file_name in inp_files:
        file_path = os.path.join(scenario_folder_path, file_name)
        # Extract step number for cleaner labels, assuming format 'Step_X.inp'
        try:
            step_num = int(file_name.split('_')[1].split('.')[0])
            time_step_labels.append(step_num)
        except (IndexError, ValueError):
            # Fallback to filename if parsing fails
            time_step_labels.append(file_name.replace('.inp', '')) 
            
        original_wn = wntr.network.WaterNetworkModel(file_path)
        original_wn.name = scenario_name # Assign a name for exclusion logic if needed
        
        # Calculate total demand for this time step
        total_demand = 0
        for node_name, node in original_wn.junctions():
            base_demand = node.base_demand
            # Handle both scalar and list base demands
            if isinstance(base_demand, list):
                for demand_item in base_demand:
                    total_demand += demand_item.base_value
            else:
                total_demand += base_demand
        total_demands.append(total_demand)

        for diameter in pipe_diameters_to_test:
            # Create a new network instance with modified diameters for each test
            modified_wn = set_pipe_diameters_for_analysis(original_wn, diameter, scenario_name)
            
            try:
                # Run the hydraulic simulation
                sim_results = run_epanet_simulation(modified_wn, static=False) # Ensure static=False for time-varying demands
                performance_metrics = evaluate_network_performance(modified_wn, sim_results)
                ds_ratio = performance_metrics['demand_satisfaction_ratio']
                results_by_diameter[diameter].append(ds_ratio)
            except Exception as e:
                print(f"  Error processing {file_name} with diameter {diameter:.4f}m: {e}")
                results_by_diameter[diameter].append(np.nan) # Use NaN for errors

    # Plotting
    if any(results_by_diameter.values()): # Check if there's anything to plot
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Primary y-axis for Demand Satisfaction Ratio
        num_diameters = len(pipe_diameters_to_test)
        colors = plt.cm.viridis(np.linspace(0, 1, num_diameters)) # Get a colormap

        for i, diameter_val in enumerate(pipe_diameters_to_test):
            ds_values = results_by_diameter[diameter_val]
            
            # Handle NaNs for plotting: Plot continuous segments
            valid_mask = ~np.isnan(ds_values)
            segments = []
            segment_start = None
            
            for j, valid in enumerate(valid_mask):
                if valid and segment_start is None:
                    segment_start = j
                elif not valid and segment_start is not None:
                    segments.append((segment_start, j - 1))
                    segment_start = None
            if segment_start is not None: # Add last segment
                segments.append((segment_start, len(valid_mask) - 1))

            label = f"Diameter {diameter_val:.4f}m"
            plotted_label = False # To ensure label is added only once per line
            
            for start_idx, end_idx in segments:
                if end_idx >= start_idx: # Ensure segment is valid
                    # Ensure x_coords match the length of y_coords segment
                    x_coords = [time_step_labels[k] for k in range(start_idx, end_idx + 1)]
                    y_coords = [ds_values[k] for k in range(start_idx, end_idx + 1)]
                    
                    if not plotted_label:
                        ax1.plot(x_coords, y_coords, label=label, marker='o', markersize=5, color=colors[i])
                        plotted_label = True
                    else:
                        ax1.plot(x_coords, y_coords, marker='o', markersize=5, color=colors[i])
        
        # Set up primary y-axis labels and limits
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Demand Satisfaction Ratio')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Create secondary y-axis for total demand
        ax2 = ax1.twinx()
        demand_line = ax2.plot(time_step_labels, total_demands, 'r--', linewidth=2, 
                              label='Total Demand', alpha=0.7)
        ax2.set_ylabel('Total Demand (m³/s)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize='small')
        
        plt.title(f'Demand Satisfaction Ratio and Total Demand vs Time Step for {scenario_name}')
        plt.tight_layout()

        # Save the plot
        save_dir = os.path.join(parent_dir, 'Plots', 'Tests')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'demand_satisfaction_analysis_{scenario_name.lower().replace(" ", "_")}.png'), dpi=300)
        print(f"  Plot saved for {scenario_name}")
        plt.show()
    else:
        print(f"No results to plot for scenario: {scenario_name}")

# A function that takes one specific scenario and determines the demand satisfaction ratio at each time step and prints

def test_ds_upd(scenario_folder, pipe_diameter):

    # Sort files in scenaio folder
    inp_files = [f for f in os.listdir(scenario_folder) if f.endswith('.inp') and f.startswith('Step_')]
    inp_files.sort(key = lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by step number
    sorted_scenario_folder = inp_files  # Update scenario_folder to only include sorted .inp files

    # print sorted scenario folder
    print(f"Sorted scenario folder: {sorted_scenario_folder}")

    for file in sorted_scenario_folder:

        if file.endswith('.inp') and file.startswith('Step_'):
            file_path = os.path.join(scenario_folder, file)
            wn = wntr.network.WaterNetworkModel(file_path)
            
            exclude_pipes = []
            if 'hanoi' in file_path.lower():
                exclude_pipes = ['1', '2'] # Example: Main source pipes for Hanoi
            elif 'anytown' in file_path.lower():
                exclude_pipes = ['4'] # Example: Main source pipe for Anytown

            for pipe_name, pipe_object in wn.pipes():
                if pipe_name not in exclude_pipes:
                    pipe_object.diameter = pipe_diameter
            
            # Run the simulation
            results = run_epanet_simulation(wn, static=False)
            performance_metrics = evaluate_network_performance(wn, results)
            ds_ratio = performance_metrics['demand_satisfaction_ratio']
            print(f"File: {file}, Demand Satisfaction Ratio: {ds_ratio:.4f}, Pipe Diameter: {pipe_diameter:.4f}m")

if __name__ == "__main__":

    net_folder_base = os.path.join(parent_dir, 'Modified_nets')

    # Scenarios to process - these are folder names within 'Modified_nets'
    scenarios_to_run = ['anytown_sprawling_3', 'hanoi_sprawling_3', 'anytown_densifying_3', 'hanoi_densifying_3']

    # Pipe diameters to test (from your 'pipes' dictionary)
    pipes_config = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    # Take a subset for clearer plots, or all of them
    diameters_for_testing = sorted(list(set(pipes_config[p]['diameter'] for p in pipes_config)))

    # for diameter in diameters_for_testing:
    #     print(f"\nTesting with pipe diameter: {diameter:.4f}m")
    #     # Take just the Sprawlgin anytown network
    #     scenario = 'anytown_sprawling_3'
    #     current_scenario_folder = os.path.join(net_folder_base, scenario)
    #     if os.path.isdir(current_scenario_folder):
    #         test_ds_upd(current_scenario_folder, diameter)
    #     else:
    #         print(f"Scenario folder '{current_scenario_folder}' does not exist. Skipping.")
    
    for scenario_name_from_list in scenarios_to_run:
        current_scenario_folder = os.path.join(net_folder_base, scenario_name_from_list)
        if os.path.isdir(current_scenario_folder):
            plot_ds_with_t_corrected(current_scenario_folder, diameters_for_testing, scenario_name_from_list)
        else:
            print(f"Scenario folder '{current_scenario_folder}' does not exist. Skipping.")