"""
This file takes a water network model (wntr) as input and displays the network, with markers outlining the reservoirs, tanks, junctions, pumps and valves. Pressure at the valves is translated to a colour map. Headlosses are displayed on the pipes and values normalised to a colour map. The network can be displayed in either 2D or 3D with the elevation of the nodes and pipes. The network is saved as a .png file in the same directory as the input file. The function also returns the figure object for further manipulation if needed.
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


def visualise_network(wn, results, title, save_path, mode='3d', show = False):
    """
    Visualise the water distribution network with pressure and headloss maps.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    results (wntr.sim.EpanetSimulator): The EPANET simulation results.
    title (str): Title of the plot.
    save_path (str): Path to save the visualisation.
    mode (str): Visualization mode, either '2d' or '3d' (default: '3d').

    Returns:
    figure: The matplotlib figure object.
    """
    # Validate mode parameter
    if mode.lower() not in ['2d', '3d']:
        raise ValueError("Mode must be either '2d' or '3d'")
    
    mode = mode.lower()
    
    # Create a figure
    figure = plt.figure(figsize=(10, 10))
    
    # Set up appropriate axes based on mode
    if mode == '3d':
        ax = figure.add_subplot(111, projection='3d')
        ax.set_zlabel('Elevation (m)')
    else:  # 2d mode
        ax = figure.add_subplot(111)
    
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Get network graph, positions, and elevations
    G = wn.get_graph()

    pos = wn.query_node_attribute('coordinates')
    # Check node positions exist

    # Extract data from results
    pressures = results.node['pressure'].iloc[0].to_dict()
    headlosses = results.link['headloss'].iloc[0].to_dict()

    # Get node lists by type
    reservoirs = wn.reservoir_name_list
    tanks = wn.tank_name_list
    junctions = wn.junction_name_list
    valves = wn.valve_name_list
    pumps = wn.pump_name_list
    pipes = wn.pipe_name_list

    # Assigns unique markers to each type of node
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

    # Override the colours of junctions with the pressure values
    pressure_norm = Normalize(vmin=np.min(list(pressures.values())), vmax=np.max(list(pressures.values())))
    pressure_cmap = plt.get_cmap('magma')
    pressure_colors = {node: pressure_cmap(pressure_norm(value)) for node, value in pressures.items()}
    
    # Create a ScalarMappable for the pressure color map
    pressure_sm = ScalarMappable(cmap=pressure_cmap, norm=pressure_norm)
    pressure_sm.set_array([])  # Only needed for colorbar
    
    # Create a colorbar for pressure
    cbar = plt.colorbar(pressure_sm, ax=ax, shrink=0.5)
    cbar.set_label('Pressure (m)')
    min_pressure = np.min(list(pressures.values()))
    max_pressure = np.max(list(pressures.values()))
    cbar.set_ticks([min_pressure, max_pressure])
    cbar.set_ticklabels([f"{min_pressure:.2f}", f"{max_pressure:.2f}"])

    # Create a color map for headlosses
    headloss_norm = Normalize(vmin=np.min(list(headlosses.values())), vmax=np.max(list(headlosses.values())))
    headloss_cmap = plt.get_cmap('plasma')
    headloss_colors = {link: headloss_cmap(headloss_norm(value)) for link, value in headlosses.items()}

    # Create a ScalarMappable for the headloss color map
    headloss_sm = ScalarMappable(cmap=headloss_cmap, norm=headloss_norm)
    headloss_sm.set_array([])  # Only needed for colorbar
    
    # Create a colorbar for headloss
    cbar_headloss = plt.colorbar(headloss_sm, ax=ax, shrink=0.5)
    cbar_headloss.set_label('Headloss (m)')
    min_headloss = np.min(list(headlosses.values()))
    max_headloss = np.max(list(headlosses.values()))
    cbar_headloss.set_ticks([min_headloss, max_headloss])
    cbar_headloss.set_ticklabels([f"{min_headloss:.2f}", f"{max_headloss:.2f}"])

    # Plot nodes with unique markers and colours
    legend_handles = []
    legend_labels = []

    for node_type, marker in node_markers.items():
        if node_type == 'reservoir':
            node_list = reservoirs
        elif node_type == 'tank':
            node_list = tanks
        elif node_type == 'junction':
            node_list = junctions
        elif node_type == 'valve':
            node_list = valves
        elif node_type == 'pump':
            node_list = pumps
        else:
            continue

        for node in node_list:
            if node in pos:
                color = pressure_colors.get(node, node_colors[node_type])
                
                if mode == '3d':
                    # 3D plotting
                    scatter = ax.scatter(
                        pos[node][0],
                        pos[node][1],
                        get_node_elevation(wn, node),
                        marker=marker,
                        color=color,
                        s=100
                    )
                else:
                    # 2D plotting
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

    # Plot pipes with headloss colors
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
            start_elev = get_node_elevation(wn, start_node)
            end_elev = get_node_elevation(wn, end_node)
            
            # Get pipe diameter for line width
            diameter = pipe_obj.diameter
            # Normalize width between 1 and 3
            width = 1 + 2 * ((diameter - min_diam) / diam_range) if diam_range != 0 else 1
            
            # Get color from headloss dictionary
            pipe_color = headloss_colors[pipe] if pipe in headloss_colors else 'gray'
            
            # Plot the pipe
            if mode == '3d':
                # 3D pipe plot
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_elev, end_elev], 
                        color=pipe_color, 
                        linewidth=width)
            else:
                # 2D pipe plot
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        color=pipe_color, 
                        linewidth=width)

    # Set the view angle for 3D mode
    if mode == '3d':
        ax.view_init(elev=30, azim=30)

    # Add legend for markers
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close(figure)
    
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
# # For 3D visualization
# visualise_network(wn, results, "Network in 3D", "network_3d.png", mode='3d')
# 
# # For 2D visualization
# visualise_network(wn, results, "Network in 2D", "network_2d.png", mode='2d')