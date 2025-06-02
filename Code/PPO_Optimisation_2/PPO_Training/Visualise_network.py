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

from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv

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


def visualise_network(wn, results, title, save_path, mode='2d', show=False, ax=None): # ADD ax=None
    """
    Visualise the water distribution network with pressure maps on a given axis.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    results (wntr.sim.EpanetSimulator): The EPANET simulation results.
    title (str): Title of the plot.
    save_path (str): Path to save the visualisation.
    mode (str): visualisation mode, either '2d' or '3d'.
    show (bool): Whether to display the plot.
    ax (matplotlib.axes.Axes, optional): The subplot axis to draw on. If None, a new figure is created.
    """
    # Validate mode parameter
    if mode.lower() not in ['2d', '3d']: #
        raise ValueError("Mode must be either '2d' or '3d'")
    
    mode = mode.lower() #
    
    # --- START OF MODIFIED SECTION for using provided 'ax' ---
    if ax is None:
        # If no axis is provided, create a new figure and axis
        figure = plt.figure(figsize=(10, 10)) #
        if mode == '3d':
            ax = figure.add_subplot(111, projection='3d') #
        else:
            ax = figure.add_subplot(111) #
    else:
        # If an axis is provided, use it and get its parent figure
        figure = ax.get_figure()
    # --- END OF MODIFIED SECTION ---
    
    ax.set_title(title) #
    ax.set_xlabel('X Coordinate') #
    ax.set_ylabel('Y Coordinate') #
    if mode == '3d':
        ax.set_zlabel('Elevation (m)') #

    pos = wn.query_node_attribute('coordinates') #

    if results:
        pressures = results.node['pressure'].iloc[0].to_dict() #
        pressure_norm = Normalize(vmin=np.min(list(pressures.values())), vmax=np.max(list(pressures.values()))) #
        pressure_cmap = plt.get_cmap('magma') #
        pressure_colors = {node: pressure_cmap(pressure_norm(value)) for node, value in pressures.items()} #
        
        pressure_sm = ScalarMappable(cmap=pressure_cmap, norm=pressure_norm) #
        pressure_sm.set_array([]) #
        
        # Use figure.colorbar to attach to the correct figure when using subplots
        cbar = figure.colorbar(pressure_sm, ax=ax, shrink=0.5, aspect=10) # Modified to use figure.colorbar
        cbar.set_label('Pressure (m)') #
        min_pressure = np.min(list(pressures.values())) #
        max_pressure = np.max(list(pressures.values())) #
        cbar.set_ticks([min_pressure, max_pressure]) #
        cbar.set_ticklabels([f"{min_pressure:.2f}", f"{max_pressure:.2f}"]) #
    else:
        pressure_colors = {} #

    node_markers = {
        'reservoir': 'o', 'tank': '^', 'junction': 's',
        'valve': 'D', 'pump': 'X'
    } #
    node_colors = {
        'reservoir': 'blue', 'tank': 'green', 'junction': 'orange',
        'valve': 'purple', 'pump': 'red'
    } #
    legend_handles = [] #
    legend_labels = [] #

    for node_type_key, marker_symbol in node_markers.items(): #
        node_list_attr = getattr(wn, f"{node_type_key}_name_list", []) #
        for node_name_val in node_list_attr: #
            if node_name_val in pos: #
                color_val = pressure_colors.get(node_name_val, node_colors[node_type_key]) #
                if mode == '3d':
                    scatter_obj = ax.scatter(
                        pos[node_name_val][0], pos[node_name_val][1], get_node_elevation(wn, node_name_val), #
                        marker=marker_symbol, color=color_val, s=100
                    ) #
                else:
                    scatter_obj = ax.scatter(
                        pos[node_name_val][0], pos[node_name_val][1], #
                        marker=marker_symbol, color=color_val, s=100
                    ) #
                if node_type_key.capitalize() not in legend_labels: #
                    legend_handles.append(scatter_obj) #
                    legend_labels.append(node_type_key.capitalize()) #

    pipe_name_list_val = wn.pipe_name_list #
    pipe_diameters_val = [wn.get_link(p_name).diameter for p_name in pipe_name_list_val] #
    min_diam_val = min(pipe_diameters_val) if pipe_diameters_val else 0 #
    max_diam_val = max(pipe_diameters_val) if pipe_diameters_val else 1 #
    diam_range_val = max_diam_val - min_diam_val #

    for p_name_val in pipe_name_list_val: #
        pipe_obj_val = wn.get_link(p_name_val) #
        start_node_name_val, end_node_name_val = pipe_obj_val.start_node_name, pipe_obj_val.end_node_name #
        if start_node_name_val in pos and end_node_name_val in pos: #
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val] #
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val) #
            diameter_val = pipe_obj_val.diameter #
            width_val = 1 + 2 * ((diameter_val - min_diam_val) / diam_range_val) if diam_range_val != 0 else 1 #
            if mode == '3d':
                ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], [start_elev_val, end_elev_val], color='gray', linewidth=width_val) #
            else:
                ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], color='gray', linewidth=width_val) #
                
    pump_line_obj = None #
    for pump_name_val in wn.pump_name_list: #
        pump_obj = wn.get_link(pump_name_val) #
        start_node_name_val, end_node_name_val = pump_obj.start_node_name, pump_obj.end_node_name #
        if start_node_name_val in pos and end_node_name_val in pos: #
            start_pos_val, end_pos_val = pos[start_node_name_val], pos[end_node_name_val] #
            start_elev_val, end_elev_val = get_node_elevation(wn, start_node_name_val), get_node_elevation(wn, end_node_name_val) #
            if mode == '3d':
                pump_line_obj = ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], [start_elev_val, end_elev_val], color='red', linewidth=2, linestyle='--', zorder=6)[0] #
            else:
                pump_line_obj = ax.plot([start_pos_val[0], end_pos_val[0]], [start_pos_val[1], end_pos_val[1]], color='red', linewidth=2, linestyle='--', zorder=6)[0] #

    if pump_line_obj is not None: #
        legend_handles.append(pump_line_obj) #
        legend_labels.append('Pump Connection') #

    if mode == '3d':
        ax.view_init(elev=30, azim=30) #

    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10) #
    plt.tight_layout() #

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') #
    if show:
        plt.show() #
    
    return figure #

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

# In Visualise_network.py

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__) #
    
    # Define paths to the INP files
    hanoi_inp_path = os.path.join(script_dir, 'Initial_networks', 'exeter', 'anytown-3.inp') #
    anytown_inp_path = os.path.join(script_dir, 'Initial_networks', 'exeter', 'hanoi-3.inp') #

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
    if results_hanoi:
        print("visualising Hanoi network in 3D...")
        visualise_network(wn_hanoi, results_hanoi, "Hanoi Network (3D)", 
                          save_path=save_hanoi, mode='3d', show=False)

    # visualise Anytown network on the second subplot
    if results_anytown:
        print("visualising Anytown network in 3D...")
        visualise_network(wn_anytown, results_anytown, "Anytown Network (3D)", 
                          save_path=save_anytown, mode='3d', show=False)
        
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad = 3.0, w_pad = 4.0) # Adjust layout to make space for suptitle and colorbars
    # # plt.savefig(os.path.join(script_dir, 'Plots', '3D_Network_visualisations.png'), dpi=150, bbox_inches='tight') #
    # plt.show()

