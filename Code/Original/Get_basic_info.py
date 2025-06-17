
"""

In this file, we get basic information about the initial WDNs in order to help determine input features such as the budget allocation with each time step

"""
import os
import numpy as np
import pandas as pd
import wntr

# anytown_network = os.path.join('Initial_networks', 'exeter', 'anytown-3.inp')
# hanoi_network = os.path.join('Initial_networks', 'exeter', 'hanoi-3.inp')

def get_basic_info(network_path):

    """
    Get basic information about the water distribution network (WDN) such as total pipe length, average pipe degree, number of junctions, reservoirs, tanks, pumps, and coordinates.
    Args:
        network_path (str): Path to the WDN input file.
    """

    wn = wntr.network.WaterNetworkModel(network_path)

    # 0 Print the name of the network
    # Extract name from the network path
    network_name = os.path.basename(network_path).split('.')[0]
    print(f"Network name: {network_name}")

    # 1. Get total pipe length    
    total_pipe_length = sum([pipe.length for pipe in wn.pipe_name_list for pipe in [wn.get_link(pipe)]])
    print(f"Total pipe length: {total_pipe_length} m")

    # 2. Get average pipe degree
    # Degree = number of connections per node
    degrees = wn.to_graph().degree()
    average_pipe_degree = sum(dict(degrees).values()) / len(wn.junction_name_list)
    print(f"Average pipe degree: {average_pipe_degree}")

    # 3. Get number of junctions
    num_junctions = len(wn.junction_name_list)
    print(f"Number of junctions: {num_junctions}")

    # 4. Get number of reservoirs
    num_reservoirs = len(wn.reservoir_name_list)
    print(f"Number of reservoirs: {num_reservoirs}")

    # 5. Get number of tanks
    num_tanks = len(wn.tank_name_list)
    print(f"Number of tanks: {num_tanks}")

    # 6. Get number of pumps
    num_pumps = len(wn.pump_name_list)
    print(f"Number of pumps: {num_pumps}")

    # 7. Get maximum and minimum coordinates of the network
    x_coords = []
    y_coords = []
    for node_name, node in wn.nodes():
        x, y = node.coordinates
        x_coords.append(x)
        y_coords.append(y)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    print(f"Min coordinates: ({min_x}, {min_y})")
    print(f"Max coordinates: ({max_x}, {max_y})")

    # 8. Get pipe diameters
    pipe_diameters = [wn.get_link(pipe_name).diameter for pipe_name in wn.pipe_name_list]
    pipe_diameters_2sf = [float(f"{d:.2g}") for d in pipe_diameters]
    print(f"Pipe diameters: {pipe_diameters_2sf} m")

    # 9. Get maximum elevation
    max_elevation = max([data.elevation for node, data in wn.junctions()])
    min_elevation = min([data.elevation for node, data in wn.junctions()])
    print(f"Maximum elevation: {max_elevation} m")
    print(f"Minimum elevation: {min_elevation} m")

if __name__ == "__main__":
    # Example usage
    anytown_network = os.path.join('PPO_Training', 'Initial_networks', 'exeter', 'anytown-3.inp')
    hanoi_network = os.path.join('PPO_Training', 'Initial_networks', 'exeter', 'hanoi-3.inp')
    get_basic_info(anytown_network)
    get_basic_info(hanoi_network)
    
    # Uncomment the following line to test with a different network
    # get_basic_info(hanoi_network)


