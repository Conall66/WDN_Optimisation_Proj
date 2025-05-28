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


def visualise_network(wn, results, title, save_path, mode='2d', show = False):
    """
    Visualise the water distribution network with pressure maps.

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
    G = wn.to_graph()

    pos = wn.query_node_attribute('coordinates')
    # Check node positions exist

    # Extract data from results
    if results:
        pressures = results.node['pressure'].iloc[0].to_dict()

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
    if results:
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
    else:
        pressure_colors = {}

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

    # Plot pipes
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
            
            # Plot the pipe with a standard gray color
            if mode == '3d':
                # 3D pipe plot
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_elev, end_elev], 
                        color='gray', 
                        linewidth=width)
            else:
                # 2D pipe plot
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        color='gray', 
                        linewidth=width)
                
    pump_line = None  # Variable to store a line object for the legend
    for pump_name in pumps:
        pump = wn.get_link(pump_name)
        start_node = pump.start_node_name
        end_node = pump.end_node_name
        
        if start_node in pos and end_node in pos:
            start_pos = pos[start_node]
            end_pos = pos[end_node]
            start_elev = get_node_elevation(wn, start_node)
            end_elev = get_node_elevation(wn, end_node)
            
            # Use a dashed red line for pumps to make them stand out
            if mode == '3d':
                pump_line = ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_elev, end_elev], 
                        color='red', 
                        linewidth=2,
                        linestyle='--',
                        zorder=6)[0]  # Get the Line2D object
                # # Add a label to identify the pump
                # mid_x = (start_pos[0] + end_pos[0]) / 2
                # mid_y = (start_pos[1] + end_pos[1]) / 2
                # mid_z = (start_elev + end_elev) / 2
                # ax.text(mid_x, mid_y, mid_z, f"Pump {pump_name}", 
                #         color='black', fontsize=8, 
                #         bbox=dict(facecolor='white', alpha=0.7),
                #         zorder=15)
            else:
                pump_line = ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        color='red', 
                        linewidth=2,
                        linestyle='--',
                        zorder=6)[0]  # Get the Line2D object
                
                # Add a label to identify the pump
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                ax.text(mid_x, mid_y, f"Pump {pump_name}", 
                        color='black', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7),
                        zorder=15)

    # Add pump line to legend if pumps exist
    if pump_line is not None:
        legend_handles.append(pump_line)
        legend_labels.append('Pump Connection')

    # Set the view angle for 3D mode
    if mode == '3d':
        ax.view_init(elev=30, azim=30)

    # Add legend for markers
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10)

    # Adjust layout and save
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
# # For 3D visualization
# visualise_network(wn, results, "Network in 3D", "network_3d.png", mode='3d')
# 
# # For 2D visualization
# visualise_network(wn, results, "Network in 2D", "network_2d.png", mode='2d')

if __name__ == "__main__":
    # Test the visualistion function without results
    # Import .inp file from modified nets
    script = os.path.dirname(__file__)
    inp_file = os.path.join(script, 'Modified_nets', 'anytown-3.inp')  # Replace with your .inp file
    wn = wntr.network.WaterNetworkModel(inp_file)
    visualise_demands(wn, "Anytown Network Demands", save_path=None, show=True)