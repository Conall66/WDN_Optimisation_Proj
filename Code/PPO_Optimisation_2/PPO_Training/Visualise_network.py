"""
This file takes a water network model (wntr) as input and displays the network, with markers outlining the reservoirs, tanks, junctions, pumps and valves. Pressure at the valves is translated to a colour map. The network can be displayed in either 2D or 3D with the elevation of the nodes and pipes. The network is saved as a .png file in the same directory as the input file. The function also returns the figure object for further manipulation if needed.
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import wntr
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from natsort import natsorted

from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

def visualise_demands(wn, title, save_path = None, show = False):
    
    # Plot network with demand values at each junction

    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Assign unique markers to each type of node
    node_markers = {
        'reservoir': 'o',
        'tank': '^',
        'junction': 's',
        'valve': 'D',
        'pump': 'X'
    }

    # Assign unique colours to each type of node
    node_colors = {
        'reservoir': 'blue',
        'tank': 'green',
        'junction': 'orange',
        'valve': 'purple',
        'pump': 'red'
    }

    # Replace the demand query with this code
    demands = {}
    for node_name, node in wn.nodes():
        if node.node_type == 'Junction':
            # Get the base demand from the first demand pattern
            if node.demand_timeseries_list:
                demands[node_name] = node.demand_timeseries_list[0].base_value * 3600
            else:
                demands[node_name] = 0.0
        else:
            # For non-junction nodes, set demand to 0
            demands[node_name] = 0.0

    # print(f"Demands: {demands}")

    # Assign node colour based on demand value
    demand_norm = Normalize(vmin=np.min(list(demands.values())), vmax=np.max(list(demands.values())))
    demand_cmap = plt.get_cmap('magma')
    demand_colors = {node: demand_cmap(demand_norm(value)) for node, value in demands.items()}
    # Create a ScalarMappable for the demand color map
    demand_sm = ScalarMappable(cmap=demand_cmap, norm=demand_norm)
    demand_sm.set_array([])  # Only needed for colorbar
    # Create a colorbar for demand
    cbar = plt.colorbar(demand_sm, ax=ax, shrink=0.5)
    cbar.set_label('Demand (m^3/h)')
    min_demand = np.min(list(demands.values()))
    max_demand = np.max(list(demands.values()))
    cbar.set_ticks([min_demand, max_demand])
    cbar.set_ticklabels([f"{min_demand:.2f}", f"{max_demand:.2f}"])

    # Covert graph to networkx graph for ease of plotting
    G = wn.to_graph()
    pos = wn.query_node_attribute('coordinates')
    # Plot nodes with unique markers and colours
    legend_handles = []
    legend_labels = []
    for node_type, marker in node_markers.items():
        if node_type == 'reservoir':
            node_list = wn.reservoir_name_list
        elif node_type == 'tank':
            node_list = wn.tank_name_list
        elif node_type == 'junction':
            node_list = wn.junction_name_list
        elif node_type == 'valve':
            node_list = wn.valve_name_list
        elif node_type == 'pump':
            node_list = wn.pump_name_list
        else:
            continue

        for node in node_list:
            if node in pos:
                color = demand_colors.get(node, node_colors[node_type])
                scatter = ax.scatter(
                    pos[node][0],
                    pos[node][1],
                    marker=marker,
                    color=color,
                    s=100
                )
                # Add to legend once per node type
                if node_type.capitalize() not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append(node_type.capitalize())

                # Add node labels
                ax.annotate(
                    node,
                    (pos[node][0], pos[node][1]),
                    textcoords="offset points",
                    xytext=(0, 5),  # Offset label slightly above the node
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

    # Plot pipes
    pipes = wn.pipe_name_list
    # Get min/max diameters for line width scaling
    pipe_diameters = [wn.get_link(pipe).diameter for pipe in pipes]
    min_diam = min(pipe_diameters) if pipe_diameters else 0
    max_diam = max(pipe_diameters) if pipe_diameters else 1
    diam_range = max_diam - min_diam
    for pipe in pipes:
        pipe_obj = wn.get_link(pipe)
        start_node = pipe_obj.start_node_name
        end_node = pipe_obj.end_node_name
        
        # Make sure the nodes exist in the position dictionary
        if start_node in pos and end_node in pos:
            start_pos = pos[start_node]
            end_pos = pos[end_node]
            # Get pipe diameter for line width
            diameter = pipe_obj.diameter
            # Normalize width between 1 and 10
            # width = 1 + 2 * ((diameter - min_diam) / diam_range) if diam_range != 0 else 1
            width = 1 + 9 * ((diameter - min_diam) / diam_range) if diam_range != 0 else 1
            
            # Plot the pipe with a standard gray color
            ax.plot([start_pos[0], end_pos[0]], 
                    [start_pos[1], end_pos[1]], 
                    color='gray', 
                    linewidth=width)
            
            ax.annotate(
                pipe,
                ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2),
                textcoords="offset points",
                xytext=(0, 5),  # Offset label slightly above the pipe
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    # Plot pump connections with a distinct appearance
    pumps = wn.pump_name_list
    pump_line = None  # Variable to store a line object for the legend
    
    for pump_name in pumps:
        pump = wn.get_link(pump_name)
        start_node = pump.start_node_name
        end_node = pump.end_node_name
        
        if start_node in pos and end_node in pos:
            start_pos = pos[start_node]
            end_pos = pos[end_node]
            
            # Use a dashed red line for pumps to make them stand out
            pump_line = ax.plot([start_pos[0], end_pos[0]], 
                      [start_pos[1], end_pos[1]], 
                      color='red', 
                      linewidth=2,
                      linestyle='--',
                      zorder=6)[0]  # Get the Line2D object
            
            # Add a label to identify the pump
            # mid_x = (start_pos[0] + end_pos[0]) / 2
            # mid_y = (start_pos[1] + end_pos[1]) / 2
            # ax.text(mid_x, mid_y, f"Pump {pump_name}", 
            #         color='black', fontsize=8, 
            #         bbox=dict(facecolor='white', alpha=0.7),
            #         zorder=15)
    
    # Add pump line to legend if pumps exist
    if pump_line is not None:
        legend_handles.append(pump_line)
        legend_labels.append('Pump Connection')
    
    # Add legend for markers
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(figure)
        print(f"Attempting to save plot to: {save_path}")
    if show:
        plt.show()
    
    return figure


def visualise_network(wn, results, title, save_path, mode='2d', show=False, ax=None, min_pressure=0):
    """
    Visualise the water distribution network with pressure deficit maps on junctions.
    Reservoirs and tanks keep their standard colors, and pipes are labeled.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    results (wntr.sim.EpanetSimulator): The EPANET simulation results.
    title (str): Title of the plot.
    save_path (str): Path to save the visualisation.
    mode (str): visualisation mode, either '2d' or '3d'.
    show (bool): Whether to display the plot.
    ax (matplotlib.axes.Axes, optional): The subplot axis to draw on. If None, a new figure is created.
    min_pressure (float): Minimum acceptable pressure. Deficits are calculated relative to this value.
    """
    # Validate mode parameter
    if mode.lower() not in ['2d', '3d']:
        raise ValueError("Mode must be either '2d' or '3d'")
    
    mode = mode.lower()
    
    if ax is None:
        # If no axis is provided, create a new figure and axis
        figure = plt.figure(figsize=(10, 10))
        if mode == '3d':
            ax = figure.add_subplot(111, projection='3d')
        else:
            ax = figure.add_subplot(111)
    else:
        # If an axis is provided, use it and get its parent figure
        figure = ax.get_figure()
    
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    if mode == '3d':
        ax.set_zlabel('Elevation (m)')

    pos = wn.query_node_attribute('coordinates')

    # Standard node colors and markers
    node_markers = {
        'reservoir': 'o', 'tank': '^', 'junction': 's',
        'valve': 'D', 'pump': 'X'
    }
    node_colors = {
        'reservoir': 'blue', 'tank': 'green', 'junction': 'orange',
        'valve': 'purple', 'pump': 'red'
    }

    # Initialize pressure colors dictionary
    pressure_colors = {}

    if results:
        # Get pressures from results
        pressures = results.node['pressure'].iloc[0].to_dict() 
        
        # Calculate pressure deficits: max(0, min_pressure - actual_pressure)
        pressure_deficits = {node: max(0, min_pressure - pressure) for node, pressure in pressures.items()}

        print(f"Pressure Deficits: {pressure_deficits}")
        
        # Set up colormap for pressure deficits (only for junctions)
        max_deficit = max(pressure_deficits.values())
        
        if max_deficit == 0:
            # No deficits - use a simple solid color
            pressure_deficit_norm = Normalize(vmin=0, vmax=1)
            pressure_deficit_cmap = plt.get_cmap('RdYlGn_r')  # Just to have a colormap defined
        else:
            # Use a colormap ranging from green (no deficit) to red (maximum deficit)
            pressure_deficit_norm = Normalize(vmin=0, vmax=max_deficit)
            pressure_deficit_cmap = plt.get_cmap('RdYlGn_r')  # Red-Yellow-Green reversed
        
        # Apply pressure colors ONLY to junctions, keep standard colors for other node types
        for node, deficit in pressure_deficits.items():
            # Check if this is a junction
            if node in wn.junction_name_list:
                if max_deficit == 0:
                    # No deficits - use a light green color for junctions
                    pressure_colors[node] = 'lightgreen'
                else:
                    # Apply colormap for junction with deficit
                    pressure_colors[node] = pressure_deficit_cmap(pressure_deficit_norm(deficit))
        
        # Create colorbar for pressure deficits
        pressure_sm = ScalarMappable(cmap=pressure_deficit_cmap, norm=pressure_deficit_norm)
        pressure_sm.set_array([])
        
        # Use figure.colorbar to attach to the correct figure when using subplots
        cbar = figure.colorbar(pressure_sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Pressure Deficit (m)')
        
        # Set colorbar ticks appropriately
        if max_deficit == 0:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["0.00 (No deficit)", "1.00 (Not used)"])
        else:
            cbar.set_ticks([0, max_deficit])
            cbar.set_ticklabels([f"0.00", f"{max_deficit:.2f}"])

    # Plot the nodes
    legend_handles = []
    legend_labels = []

    for node_type_key, marker_symbol in node_markers.items():
        node_list_attr = getattr(wn, f"{node_type_key}_name_list", [])
        for node_name_val in node_list_attr:
            if node_name_val in pos:
                # Use pressure colors for junctions if available, otherwise standard color
                if node_type_key == 'junction' and node_name_val in pressure_colors:
                    color_val = pressure_colors[node_name_val]
                else:
                    # For reservoirs, tanks and other nodes, always use standard color
                    color_val = node_colors[node_type_key]
                
                if mode == '3d':
                    scatter_obj = ax.scatter(
                        pos[node_name_val][0], pos[node_name_val][1], get_node_elevation(wn, node_name_val),
                        marker=marker_symbol, color=color_val, s=100
                    )
                else:
                    scatter_obj = ax.scatter(
                        pos[node_name_val][0], pos[node_name_val][1],
                        marker=marker_symbol, color=color_val, s=100
                    )
                
                # Add node labels (optional - similar to your visualise_demands function)
                ax.annotate(
                    node_name_val,
                    (pos[node_name_val][0], pos[node_name_val][1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
                
                if node_type_key.capitalize() not in legend_labels:
                    legend_handles.append(scatter_obj)
                    legend_labels.append(node_type_key.capitalize())

    # Plot pipes with labels
    pipe_name_list_val = wn.pipe_name_list
    pipe_diameters_val = [wn.get_link(p_name).diameter for p_name in pipe_name_list_val]
    min_diam_val = min(pipe_diameters_val) if pipe_diameters_val else 0
    max_diam_val = max(pipe_diameters_val) if pipe_diameters_val else 1
    diam_range_val = max_diam_val - min_diam_val

    for p_name_val in pipe_name_list_val:
        pipe_obj_val = wn.get_link(p_name_val)
        start_node_name_val, end_node_name_val = pipe_obj_val.start_node_name, pipe_obj_val.end_node_name
        
        if start_node_name_val in pos and end_node_name_val in pos:
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val]
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val)
            diameter_val = pipe_obj_val.diameter
            width_val = 1 + 2 * ((diameter_val - min_diam_val) / diam_range_val) if diam_range_val != 0 else 1
            
            # Plot the pipe line
            if mode == '3d':
                ax.plot([start_pos_val[0], end_pos_val[0]], 
                        [start_pos_val[1], end_pos_val[1]], 
                        [start_elev_val, end_elev_val], 
                        color='gray', 
                        linewidth=width_val)
            else:
                ax.plot([start_pos_val[0], end_pos_val[0]], 
                        [start_pos_val[1], end_pos_val[1]], 
                        color='gray', 
                        linewidth=width_val)
            
            # Add pipe label
            # Calculate midpoint for label placement
            mid_x = (start_pos_val[0] + end_pos_val[0]) / 2
            mid_y = (start_pos_val[1] + end_pos_val[1]) / 2
            
            ax.annotate(
                p_name_val,
                (mid_x, mid_y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
                
    # Plot pumps
    pump_line_obj = None
    for pump_name_val in wn.pump_name_list:
        pump_obj = wn.get_link(pump_name_val)
        start_node_name_val, end_node_name_val = pump_obj.start_node_name, pump_obj.end_node_name
        
        if start_node_name_val in pos and end_node_name_val in pos:
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val]
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val)
            
            if mode == '3d':
                pump_line_obj = ax.plot([start_pos_val[0], end_pos_val[0]], 
                                       [start_pos_val[1], end_pos_val[1]], 
                                       [start_elev_val, end_elev_val], 
                                       color='red', 
                                       linewidth=2, 
                                       linestyle='--', 
                                       zorder=6)[0]
            else:
                pump_line_obj = ax.plot([start_pos_val[0], end_pos_val[0]], 
                                       [start_pos_val[1], end_pos_val[1]], 
                                       color='red', 
                                       linewidth=2, 
                                       linestyle='--', 
                                       zorder=6)[0]
            
            # Add pump label
            mid_x = (start_pos_val[0] + end_pos_val[0]) / 2
            mid_y = (start_pos_val[1] + end_pos_val[1]) / 2
            ax.annotate(
                pump_name_val,
                (mid_x, mid_y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    if pump_line_obj is not None:
        legend_handles.append(pump_line_obj)
        legend_labels.append('Pump Connection')

    if mode == '3d':
        ax.view_init(elev=30, azim=30)

    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    return figure

def get_node_elevation(wn, node_name):
    """
    Get the elevation of a node in the water network model.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    node_name (str): The name of the node.

    Returns:
    float: The elevation of the node.
    """
    if node_name in wn.junction_name_list:
        return wn.get_node(node_name).elevation
    elif node_name in wn.tank_name_list:
        return wn.get_node(node_name).elevation
    elif node_name in wn.reservoir_name_list:
        return wn.get_node(node_name).base_head
    elif node_name in wn.valve_name_list:
        return wn.get_node(node_name).elevation
    elif node_name in wn.pump_name_list:
        return wn.get_node(node_name).elevation
    else:
        return 0


# Example usage:
# import wntr
# wn = wntr.network.WaterNetworkModel('example.inp')
# sim = wntr.sim.EpanetSimulator(wn)
# results = sim.run_sim()
# 
# # For 3D visualisation
# visualise_network(wn, results, "Network in 3D", "network_3d.png", mode='3d')
# 
# # For 2D visualisation
# visualise_network(wn, results, "Network in 2D", "network_2d.png", mode='2d')

def visualise_scenario(model_path: str, scenario_name: str, time_step_to_visualise: int):
    """
    Runs a trained agent on a specific scenario and visualises its upgrade decisions.

    Args:
        model_path (str): Path to the trained agent model file.
        scenario_name (str): The name of the scenario to run (e.g., 'anytown_sprawling_3').
        time_step_to_visualise (int): The specific time step within the scenario to visualise (e.g., 5).
    """
    print(f"\n--- visualising Agent Decisions ---")
    print(f"Model: {model_path}")
    print(f"Scenario: {scenario_name}, Time Step: {time_step_to_visualise}")

    # --- 1. Load Configurations and Agent ---
    pipes_config = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]

    env = WNTRGymEnv(pipes_config, scenarios)
    vec_env = DummyVecEnv([lambda: env])
    agent = GraphPPOAgent(vec_env, pipes_config)
    agent.load(model_path)

    # --- 2. Run the Scenario to the Desired Time Step ---
    obs, info = env.reset(scenario_name=scenario_name)

    # Fast-forward to the target time step
    while env.current_time_step < time_step_to_visualise:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        if info.get('pipe_changes') is not None:
             print(f"  > Completed time step {env.current_time_step - 1}...")

    if env.current_time_step != time_step_to_visualise:
        print(f"Error: Could not reach time step {time_step_to_visualise}. The scenario may be shorter.")
        return

    print(f"\nAt Time Step {time_step_to_visualise}. Running agent to record decisions...")

    # --- 3. Record Agent's Decisions for the Current Time Step ---
    upgraded_pipe_details = {}
    network_before_changes = env.current_network.copy()
    
    # Loop until the agent has made a decision for every pipe in this time step
    done = False
    while not done and info.get('pipe_changes') is None:
        pipe_name = env.pipe_names[env.current_pipe_index]
        action, _ = agent.predict(obs, deterministic=True)

        if action > 0: # Action > 0 is an upgrade
            new_diameter = env.pipe_diameter_options[action - 1]
            upgraded_pipe_details[pipe_name] = new_diameter
            print(f"  - Agent chose to UPGRADE pipe '{pipe_name}' to {new_diameter:.4f} m")

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # --- 4. Prepare for Plotting ---
    # Use the network state *before* the agent's changes for the visualisation base
    wn = network_before_changes
    link_colors = {}
    link_widths = {}
    
    # Set default colors and widths
    for pipe_name in wn.pipe_name_list:
        link_colors[pipe_name] = 'lightgray'
        link_widths[pipe_name] = 1.5

    # Highlight upgraded pipes
    for pipe_name in upgraded_pipe_details:
        link_colors[pipe_name] = 'crimson' # Color for upgraded pipes
        link_widths[pipe_name] = 4.0      # Thicker line for upgraded pipes

    # --- 5. Generate and Save the visualisation ---
    fig, ax = plt.subplots(figsize=(15, 12))
    wntr.graphics.plot_network(
        wn,
        ax=ax,
        node_size=40,
        link_color=link_colors,
        link_width=link_widths,
        title=f"Agent Decisions for {scenario_name} | Time Step {time_step_to_visualise}"
    )

    # Create a custom legend
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2, label='Unchanged Pipe'),
        Line2D([0], [0], color='crimson', lw=4, label='Upgraded Pipe')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='large')

    # Save the plot
    plots_dir = "Plots/Agent_Decisions"
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"{scenario_name}_T{time_step_to_visualise}_Decisions.png")
    plt.savefig(save_path)
    print(f"\nvisualisation saved to: {save_path}")
    plt.show()

# Add this new function to your Visualise_network.py file

def visualise_pipe_diameters(wn, title, save_path=None, mode='2d', show=False, ax=None):
    """
    Visualise the water distribution network with pipe diameters color-coded and width-scaled.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    title (str): Title of the plot.
    save_path (str, optional): Path to save the visualisation. Defaults to None.
    mode (str, optional): Visualisation mode, either '2d' or '3d'. Defaults to '2d'.
    show (bool, optional): Whether to display the plot. Defaults to False.
    ax (matplotlib.axes.Axes, optional): The subplot axis to draw on. If None, a new figure is created.
    """
    if mode.lower() not in ['2d', '3d']:
        raise ValueError("Mode must be either '2d' or '3d'")
    mode = mode.lower()

    if ax is None:
        figure = plt.figure(figsize=(12, 10)) # Adjusted size for better legend/colorbar spacing
        if mode == '3d':
            ax = figure.add_subplot(111, projection='3d')
        else:
            ax = figure.add_subplot(111)
    else:
        figure = ax.get_figure()

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    if mode == '3d':
        ax.set_zlabel('Elevation (m)')

    pos = wn.query_node_attribute('coordinates')

    # --- Node Plotting (Simplified - focus is on pipes) ---
    node_markers = {
        'reservoir': 'o', 'tank': '^', 'junction': 's',
        'valve': 'D', 'pump': 'X'
    }
    node_colors_standard = { # Standard colors for nodes when not showing pressure
        'reservoir': 'blue', 'tank': 'green', 'junction': 'gray',
        'valve': 'purple', 'pump': 'red'
    }
    legend_handles = []
    legend_labels = []

    for node_type_key, marker_symbol in node_markers.items():
        node_list_attr = getattr(wn, f"{node_type_key}_name_list", [])
        # Only plot nodes that have coordinates
        nodes_to_plot = [n for n in node_list_attr if n in pos]
        if not nodes_to_plot:
            continue

        # Get coordinates and elevations for all nodes of this type at once
        x_coords = [pos[n][0] for n in nodes_to_plot]
        y_coords = [pos[n][1] for n in nodes_to_plot]
        
        color_val = node_colors_standard.get(node_type_key, 'black') # Default color

        if mode == '3d':
            z_coords = [get_node_elevation(wn, n) for n in nodes_to_plot]
            scatter_obj = ax.scatter(x_coords, y_coords, z_coords,
                                     marker=marker_symbol, color=color_val, s=50, alpha=0.7, label=node_type_key.capitalize())
        else:
            scatter_obj = ax.scatter(x_coords, y_coords,
                                     marker=marker_symbol, color=color_val, s=50, alpha=0.7, label=node_type_key.capitalize())
        
        if node_type_key.capitalize() not in legend_labels:
            # For legend, create a dummy scatter if using ax.legend later with handles/labels from loop
            # Or, rely on the label in ax.scatter and just call ax.legend() once
            pass # Legend will be handled by ax.legend() if labels are set in scatter

    # --- Pipe Diameter Visualisation ---
    pipe_name_list_val = wn.pipe_name_list
    if not pipe_name_list_val:
        print("No pipes in the network to visualize.")
        return figure

    pipe_diameters_map = {p_name: wn.get_link(p_name).diameter for p_name in pipe_name_list_val}
    all_diameters = list(pipe_diameters_map.values())
    
    if not all_diameters:
        min_diam_val = 0
        max_diam_val = 1 # Avoid division by zero if all_diameters is empty
    else:
        min_diam_val = min(all_diameters)
        max_diam_val = max(all_diameters)

    diam_range_val = max_diam_val - min_diam_val
    if diam_range_val == 0: # Handle case where all pipes have the same diameter
        diam_range_val = 1 

    # Setup colormap for pipe diameters
    # Using 'cividis' as it's perceptually uniform and good for colorblindness
    pipe_diam_norm = Normalize(vmin=min_diam_val, vmax=max_diam_val)
    pipe_diam_cmap = plt.get_cmap('viridis') 

    for p_name_val in pipe_name_list_val:
        pipe_obj_val = wn.get_link(p_name_val)
        start_node_name_val, end_node_name_val = pipe_obj_val.start_node_name, pipe_obj_val.end_node_name
        
        if start_node_name_val in pos and end_node_name_val in pos:
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val]
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val)
            
            diameter_val = pipe_diameters_map[p_name_val]
            
            # Scale line width by diameter (adjust multiplier for desired effect)
            width_val = 0.5 + 4 * ((diameter_val - min_diam_val) / diam_range_val)
            
            # Get color based on diameter
            pipe_color = pipe_diam_cmap(pipe_diam_norm(diameter_val))
            
            if mode == '3d':
                ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], [start_elev_val, end_elev_val],
                        color=pipe_color, linewidth=width_val, alpha=0.8, zorder=1)
            else:
                ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]],
                        color=pipe_color, linewidth=width_val, alpha=0.8, zorder=1)

    # Add colorbar for pipe diameters
    sm_diam = ScalarMappable(cmap=pipe_diam_cmap, norm=pipe_diam_norm)
    sm_diam.set_array([]) # Important for the colorbar to show up
    cbar_diam = figure.colorbar(sm_diam, ax=ax, shrink=0.6, aspect=20, label='Pipe Diameter (m)', orientation='vertical')
    # Add ticks for min, max, and mid if possible
    cbar_diam_ticks = [min_diam_val, (min_diam_val + max_diam_val) / 2, max_diam_val]
    if len(set(cbar_diam_ticks)) < 2 : # if min and max are too close or same
        cbar_diam_ticks = np.unique([min_diam_val, max_diam_val])
    cbar_diam.set_ticks(cbar_diam_ticks)
    cbar_diam.ax.tick_params(labelsize=8)


    # --- Pump Plotting (as before, simple lines) ---
    pump_line_obj = None
    for pump_name_val in wn.pump_name_list:
        pump_obj = wn.get_link(pump_name_val)
        start_node_name_val, end_node_name_val = pump_obj.start_node_name, pump_obj.end_node_name
        if start_node_name_val in pos and end_node_name_val in pos:
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val]
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val)
            line_color_pump = 'magenta' # Distinct color for pumps
            if mode == '3d':
                pump_line_obj_temp = ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], [start_elev_val, end_elev_val],
                                       color=line_color_pump, linewidth=2, linestyle='--', zorder=2)[0]
            else:
                pump_line_obj_temp = ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]],
                                       color=line_color_pump, linewidth=2, linestyle='--', zorder=2)[0]
            if pump_line_obj is None: pump_line_obj = pump_line_obj_temp # For legend handle

    # Create legend for node types and pump lines
    # Node legend handles come from the scatter plots if 'label' was used
    current_handles, current_labels = ax.get_legend_handles_labels()
    if pump_line_obj is not None and 'Pump Connection' not in current_labels:
        current_handles.append(pump_line_obj)
        current_labels.append('Pump Connection')
    
    if current_handles: # Only show legend if there's something to show
        ax.legend(current_handles, current_labels, loc='upper right', fontsize=10)

    if mode == '3d':
        ax.view_init(elev=20, azim=45) # Adjusted view for potentially better visibility

    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust rect to make space for colorbar

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Diameter visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving diameter plot: {e}")
            
    if show:
        plt.show()
    
    return figure

# Add this function to your Visualise_network.py file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # For easier data manipulation for the heatmap
import os
# Make sure these can be imported from the context where Visualise_network.py is located
# You might need to adjust relative paths if Visualise_network.py is in a different directory
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv

def plot_pipe_diameters_heatmap_over_time(
    model_path: str,
    pipes_config: dict,
    scenarios_list: list, # Full list of scenarios for environment initialization
    target_scenario_name: str,
    num_episodes_for_data: int = 1, # Number of episodes to run for collecting data
    save_dir: str = "Plots/Pipe_Diameter_Evolution"
):
    """
    Runs a trained agent on a scenario, records pipe diameters at each major time step
    for all pipes, and plots this data as a heatmap.

    Args:
        model_path (str): Path to the trained agent model file (without .zip).
        pipes_config (dict): The pipes configuration dictionary.
        scenarios_list (list): List of all possible scenarios (for env initialization).
        target_scenario_name (str): The specific scenario to run and collect data from.
        num_episodes_for_data (int): Number of episodes to average data over.
                                     For a single representative run, use 1.
        save_dir (str): Directory to save the generated plot.
    """
    print(f"\n--- Generating Pipe Diameter Evolution Heatmap ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Target Scenario: {target_scenario_name}")
    print(f"Collecting data over {num_episodes_for_data} episode(s).")

    # --- 1. Setup Environment and Agent ---
    # We use the base WNTRGymEnv directly to step through and inspect.
    env = WNTRGymEnv(
        pipes=pipes_config, 
        scenarios=scenarios_list,
        # Pass the static estimates for normalization if you're using them
        # current_max_pd=5000.0, # Example value, ensure it matches your training
        # current_max_cost=2.5e7 # Example value
    )
    
    # Agent needs a VecEnv for its constructor, even if it's a Dummy one.
    # temp_vec_env = DummyVecEnv([lambda: WNTRGymEnv(pipes_config, scenarios_list)])

    temp_vec_env = SubprocVecEnv([lambda: WNTRGymEnv(pipes_config, scenarios_list)], start_method='spawn')

    agent = GraphPPOAgent(temp_vec_env, pipes_config=pipes_config)
    agent.load(model_path)
    temp_vec_env.close() # We'll use the 'env' instance directly.

    # --- 2. Data Collection ---
    # This list will store DataFrames, each from one episode.
    # Each DataFrame will have time steps as index and pipe IDs as columns.
    all_episodes_dfs = []

    for episode_idx in range(num_episodes_for_data):
        print(f"  Running Episode {episode_idx + 1}/{num_episodes_for_data} for data collection...")
        obs, info = env.reset(scenario_name=target_scenario_name)
        
        # {time_step: {pipe_id: diameter}}
        episode_diameter_data_dict = {} 
        
        done = False
        major_step_count = 0

        while not done:
            # Record diameters at the START of the current major_step_count
            current_diameters_at_step = {}
            if env.current_network: # Ensure network is loaded
                for pipe_name in env.pipe_names: # env.pipe_names is updated in _load_network_for_timestep
                    current_diameters_at_step[pipe_name] = env.current_network.get_link(pipe_name).diameter
            if current_diameters_at_step: # Only add if there's data
                 episode_diameter_data_dict[major_step_count] = current_diameters_at_step
            
            # Let the agent make decisions for all pipes in this major step
            # The WNTRGymEnv's step method internally loops through current_pipe_index
            # until a full set of decisions is made or the episode ends.
            actions_for_this_major_step_completed = False
            while not actions_for_this_major_step_completed and not done:
                batched_obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
                action_int, _ = agent.predict(batched_obs, deterministic=True)
                
                obs, reward, terminated, truncated, step_info = env.step(action_int.item())
                done = terminated or truncated

                # Check if the 'step' completed a full pass over pipes
                if step_info.get('pipe_changes') is not None or done:
                    actions_for_this_major_step_completed = True
            
            if not done:
                major_step_count += 1 # Increment for the next state to be recorded

        if episode_diameter_data_dict:
            # Convert this episode's data to a DataFrame: Index=time_step, Columns=pipe_id
            episode_df = pd.DataFrame.from_dict(episode_diameter_data_dict, orient='index')
            all_episodes_dfs.append(episode_df)
    
    env.close()

    if not all_episodes_dfs:
        print("No diameter data collected. Cannot generate heatmap.")
        return None

    # --- 3. Process Data for Heatmap ---
    # If multiple episodes, average the DataFrames
    if len(all_episodes_dfs) > 1:
        # Concatenate along axis 0 (rows are time steps), then group by time step and average.
        # This handles cases where some pipes might not appear in all episodes/timesteps (NaNs).
        processed_data_df = pd.concat(all_episodes_dfs).groupby(level=0).mean()
    elif len(all_episodes_dfs) == 1:
        processed_data_df = all_episodes_dfs[0]
    else: # Should be caught above
        return None

    # We want pipes as rows (Y-axis) and time steps as columns (X-axis)
    heatmap_df = processed_data_df.transpose()

    # Sort Pipe IDs for consistent Y-axis order (optional, but good for readability)
    # Natural sort would be better if pipe names are like 'P1', 'P10', 'P2'.
    # For now, simple alphabetical sort.
    try:
        sorted_pipe_ids = natsorted(heatmap_df.index)
    except ImportError:
        print("natsort library not found, using simple sort for pipe IDs.")
        sorted_pipe_ids = sorted(heatmap_df.index)

    print(f"Sorted Pipe IDs: {sorted_pipe_ids}")
    
    heatmap_df = heatmap_df.reindex(sorted_pipe_ids)

    if heatmap_df.empty:
        print("Processed heatmap data is empty after processing. Cannot plot.")
        return None

    # --- 4. Plotting the Heatmap ---
    fig_height = max(8, len(heatmap_df.index) * 0.1) # Adjust height based on number of pipes
    fig_width = max(12, heatmap_df.shape[1] * 0.3)   # Adjust width based on number of time steps
    
    plt.figure(figsize=(fig_width, fig_height))
    
    try:
        import seaborn as sns
        ax = sns.heatmap(
            heatmap_df, 
            annot=False, # Annotations can be cluttered; consider for smaller number of pipes/steps
            cmap="viridis", # Colormap (e.g., "viridis", "plasma", "magma")
            linewidths=.5,
            cbar_kws={'label': 'Pipe Diameter (m)'}
        )
    except ImportError:
        print("Seaborn not found. Using basic matplotlib imshow for heatmap.")
        plt.imshow(heatmap_df, aspect='auto', cmap="viridis", interpolation='nearest')
        plt.colorbar(label='Pipe Diameter (m)')
        ax = plt.gca() # Get current axes for labeling

    ax.set_title(f"Pipe Diameter Evolution: {target_scenario_name}\n(Agent: {os.path.basename(model_path)}, {num_episodes_for_data} Episode(s) Data)", fontsize=14)
    ax.set_xlabel("Major Environment Time Step", fontsize=12)
    ax.set_ylabel("Pipe ID", fontsize=12)
    
    # Ensure all time steps are shown as ticks if not too many
    if heatmap_df.shape[1] < 30: # If fewer than 30 time steps, show all
         ax.set_xticks(np.arange(heatmap_df.shape[1]) + 0.5)
         ax.set_xticklabels(heatmap_df.columns)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0) # Keep pipe IDs horizontal
    
    plt.tight_layout()

    # --- 5. Save and Show ---
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = f"diam_heatmap_{os.path.basename(model_path)}_{target_scenario_name}.png"
    full_save_path = os.path.join(save_dir, plot_filename)
    try:
        plt.savefig(full_save_path)
        print(f"Pipe diameter heatmap saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")

    plt.show()
    plt.close(plt.gcf()) # Close the figure to free memory
    
    return plt.gcf() # Return the figure object

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__) #
    
    # Define paths to the INP files
    # hanoi_inp_path = os.path.join(script_dir, 'Initial_networks', 'exeter', 'hanoi-3.inp') #
    # anytown_inp_path = os.path.join(script_dir, 'Initial_networks', 'exeter', 'anytown-3.inp') #

    hanoi_inp_path = os.path.join(script_dir, 'Modified_nets', 'hanoi-3.inp') #
    anytown_inp_path = os.path.join(script_dir, 'Modified_nets', 'anytown-3.inp') #

    # Load the network models
    wn_hanoi = wntr.network.WaterNetworkModel(hanoi_inp_path) #
    wn_anytown = wntr.network.WaterNetworkModel(anytown_inp_path) #

    # Run simulations to get results (for pressure visualisation)
    print("Running simulation for Hanoi network...")
    results_hanoi = run_epanet_simulation(wn_hanoi) #
    print("Running simulation for Anytown network...")
    results_anytown = run_epanet_simulation(wn_anytown) #

    script = os.path.dirname(os.path.abspath(__file__)) #
    save_hanoi = os.path.join(script, 'Plots', 'Hanoi_net_3d.png')
    save_anytown = os.path.join(script, 'Plots', 'Anytown_net_3d.png')

    # Create a figure with two 3D subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), subplot_kw={'projection': '3d'})
    # fig.suptitle('3D Network visualisations with Pressure', fontsize=16)

    # visualise Hanoi network on the first subplot
    # if results_hanoi:
    #     print("visualising Hanoi network in 3D...")
    #     visualise_network(wn_hanoi, results_hanoi, "Hanoi Network (3D)", 
    #                       save_path=save_hanoi, mode='3d', show=False)

    # # visualise Anytown network on the second subplot
    # if results_anytown:
    #     print("visualising Anytown network in 3D...")
    #     visualise_network(wn_anytown, results_anytown, "Anytown Network (3D)", 
    #                       save_path=save_anytown, mode='3d', show=False)
        
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad = 3.0, w_pad = 4.0) # Adjust layout to make space for suptitle and colorbars
    # # plt.savefig(os.path.join(script_dir, 'Plots', '3D_Network_visualisations.png'), dpi=150, bbox_inches='tight') #
    # plt.show()

    # visualise_demands(wn_anytown, "Anytown Network Demands", save_path=os.path.join(script, 'Plots', 'Anytown_demands.png'), show=True)

    # Generate a heatmap from a pretrained agent
    model_path = os.path.join(script_dir, 'agents', 'agent1_hanoi_only_20250603_064211.zip')

    pipes_config = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    scenarios_list = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 
        'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 
        'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]

    # target_scenario_name = 'hanoi_densifying_3' # Example scenario to visualize
    # num_episodes_for_data = 4 # For a single representative run, use 1
    # plot_pipe_diameters_heatmap_over_time(
    #     model_path=model_path,
    #     pipes_config=pipes_config,
    #     scenarios_list=scenarios_list,
    #     target_scenario_name=target_scenario_name,
    #     num_episodes_for_data=num_episodes_for_data,
    #     save_dir=os.path.join(script_dir, 'Plots', 'Pipe_Diameter_Evolution')
    # )

    # Plot the pressure visualisation for both networks but with pressure deficit
    visualise_network(wn_hanoi, results_hanoi, "Hanoi Network with Pressure Deficit", 
                      save_path=os.path.join(script, 'Plots', 'Hanoi_net_pressure_deficit.png'), mode='2d', show=True)
    visualise_network(wn_anytown, results_anytown, "Anytown Network with Pressure Deficit",
                      save_path=os.path.join(script, 'Plots', 'Anytown_net_pressure_deficit.png'), mode='2d', show=True)

