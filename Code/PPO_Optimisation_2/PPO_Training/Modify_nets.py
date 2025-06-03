
"""

In this file, we take the initial networks and convert units to match. we then assign pipe diameters closest to those of the originial network from the discrete set

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from shutil import copyfile
import random
import wntr
from wntr.network.io import write_inpfile
import time
from wntr.graphics.network import plot_network

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Visualise_network import visualise_network, visualise_demands
from Elevation_map import generate_elevation_map

# In Modify_nets.py

# Import necessary libraries
import os
import wntr
from wntr.network.io import write_inpfile
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# The convert_units and map_pipe_diameters functions are correct and do not need changes.
def convert_units(wn):
    units = wn.options.hydraulic.inpfile_units
    if units != 'CMH':
        wn.options.hydraulic.inpfile_units = 'CMH'
        units = 'CMH'
    return wn, units

def map_pipe_diameters(wn, discrete_diameters):
    for pipe_id, pipe in wn.pipes():
        original_diameter = pipe.diameter
        closest_diameter = min(discrete_diameters, key=lambda x: abs(x - original_diameter))
        pipe.diameter = closest_diameter
    return wn

def enable_pumps(wn):
    """
    This function adds time-based CONTROLS for each pump to modulate their
    speed according to a predefined pattern. This is the robust method
    that ensures the logic is written to the final .inp file.
    """
    # print("Enabling pumps with time-based speed controls...")
    pump_pattern = [0.8, 0.7, 0.7, 0.6, 0.6, 0.7,  # Hours 0-5
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # Hours 6-11
                    1.2, 1.2, 1.2, 1.3, 1.3, 1.2,   # Hours 12-17
                    1.1, 1.0, 1.0, 0.9, 0.8, 0.8]   # Hours 18-23
    
    wn.add_pattern('pump_pattern', pump_pattern)

    # Loop through each pump in the network
    pump_name = wn.pump_name_list[0]
    pump_1 = wn.get_link(pump_name)
    pump_1.speed_pattern_name = 'pump_pattern'

    # Add a base-speed to pump 1
    pump_1.base_speed = 1.0  # Set the base speed to 1.0 (100%)
    
    return wn

def convert_reservoir_to_pump(wn, reservoir_name='1'):
    """
    Converts an elevated reservoir to a zero-elevation reservoir with a power pump.
    
    Parameters:
    -----------
    wn : WaterNetworkModel
        The water network model to modify
    reservoir_name : str
        The name of the reservoir to convert (default is '1' for Hanoi)
        
    Returns:
    --------
    wn : WaterNetworkModel
        The modified water network model
    """
    # Get the reservoir and its original elevation
    reservoir = wn.get_node(reservoir_name)
    original_elevation = reservoir.base_head
    
    # Store the original head to maintain hydraulic equivalence
    original_head = original_elevation
    
    # Set reservoir elevation to 0
    reservoir.base_head = 0.0
    print(f"Set reservoir {reservoir_name} elevation from {original_elevation} to 0.0")
    
    # Find the pipe connected to the reservoir
    connected_links = []
    for link_name, link in wn.links():
        if link.start_node_name == reservoir_name or link.end_node_name == reservoir_name:
            connected_links.append(link_name)
    
    if not connected_links:
        raise ValueError(f"No links found connected to reservoir {reservoir_name}")
    
    # Create a pump to replace the head provided by the elevation
    # Using the first connected pipe to determine where to place the pump
    pipe_to_reservoir = connected_links[0]
    
    # Get pipe's start and end nodes
    pipe = wn.get_link(pipe_to_reservoir)
    
    # Calculate approximate power based on head
    # P = ρ * g * Q * H / (η * 1000)
    # where ρ = 1000 kg/m³, g = 9.81 m/s², η = 0.75 (efficiency)
    # We'll use a design flow of 500 m³/h = 0.139 m³/s
    design_flow = 500  # m³/s (500 m³/h)
    density = 1000  # kg/m³
    gravity = 9.81  # m/s²
    efficiency = 0.75
    
    # Calculate power in kW
    power = (density * gravity * design_flow * original_head) / (efficiency * 1000)
    
    # Create a power curve
    # curve_name = 'PUMP_POWER_CURVE'
    # # Single point power curve (flow, power in kW)
    # wn.add_curve(curve_name, 'POWER', [(0, power)])
    
    # Delete the pipe and add a pump in its place
    start_node = pipe.start_node_name
    end_node = pipe.end_node_name
    
    # Remove the original pipe
    wn.remove_link(pipe_to_reservoir)
    
    # Add the power pump
    pump_name = 'PUMP_' + pipe_to_reservoir
    wn.add_pump(pump_name, 
                start_node_name=start_node, 
                end_node_name=end_node, 
                pump_type='POWER')
    wn.get_link(pump_name).power = power
    print(f"Added power pump '{pump_name}' between nodes {start_node} and {end_node} with power {power:.2f} kW (equivalent to {original_head}m head)")
    
    return wn

# Map to elevation profile - Assign elevation profile to Hanoi network for more realistic network optimisation
def assign_elevation_profile(wn):
    # Generates an elevation profile for the area space and assigns elevation values to each of the nodes in the network
    
    # Get max and min coordinates of the network
    x_coords = [node_data.coordinates[0] for node, node_data in wn.nodes()]
    y_coords = [node_data.coordinates[1] for node, node_data in wn.nodes()]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    area_size = (int(max_x - min_x), int(max_y - min_y))
    print(f"Area size: {area_size}")

    elevation_range = (0.0, 100.0)  # Define the elevation range for the map

    # Generate elevation map
    elevation_map, peaks = generate_elevation_map(area_size=area_size,
                                                  elevation_range=elevation_range, 
                                                  num_peaks=3, 
                                                  landscape_type='hilly')
    # Assign elevation values to nodes based on their coordinates
    for node, node_data in wn.junctions():
        x, y = node_data.coordinates
        # Normalize coordinates to match elevation map dimensions
        norm_x = int((x - min_x) / area_size[0] * (elevation_map.shape[0] - 1))
        norm_y = int((y - min_y) / area_size[1] * (elevation_map.shape[1] - 1))
        node_data.base_head = elevation_map[norm_x, norm_y]

    for node, node_data in wn.reservoirs():
        node_data.base_head = max(elevation_range)

    for node, node_data in wn.tanks():
        node_data.base_head = max(elevation_range) / 2

    print("Assigned elevation profile to nodes in the network.")

    return wn

if __name__ == "__main__":
    
    files = ['anytown-3.inp', 'hanoi-3.inp']
    script = os.path.dirname(__file__)

    print("----------------------------------")

    for file in files:
        file_path = os.path.join(script, 'Initial_networks', 'exeter', file)
        save_path = os.path.join(script, 'Modified_nets', file)
        
        print(f"\nProcessing network: {file}")

        # 1. Load the model from the original file
        wn = wntr.network.WaterNetworkModel(file_path)
    
        # 2. Convert units in memory
        wn, units = convert_units(wn)

        # 4. Map pipe diameters in memory
        print("Mapping pipe diameters...")
        discrete_diameters = [0.3048, 0.4064, 0.508, 0.609, 0.762, 1.016]
        # wn = map_pipe_diameters(wn, discrete_diameters)

        # 3. Apply modifications specific to each network
        if 'anytown' in file.lower():
            name = 'Anytown'
            # Enable pumps for Anytown
            wn = enable_pumps(wn)
            exclude_pipes = ['4', '33', '40', '142', '143']
            for pipe, pipe_data in wn.pipes():
                if pipe_data.name not in exclude_pipes:
                    pipe_data.diameter = min(discrete_diameters)
            
        elif 'hanoi' in file.lower():
            # For Hanoi, convert the reservoir to pump setup
            # wn = convert_reservoir_to_pump(wn, reservoir_name='1')
            # # Assign elevation profile to Hanoi network
            # wn = assign_elevation_profile(wn)

            # Add a 50kWh pump
            # pump_name = 'PUMP_1'
            # wn.add_pump(pump_name,
            #             start_node_name='5', 
            #             end_node_name='6', 
            #             pump_type='POWER',
            #             pump_parameter = 50.0)
            name = 'Hanoi'

            exclude_pipes = ['12', '11', '10', '2', '1', '21', '22']
            for pipe, pipe_data in wn.pipes():
                if pipe_data.name not in exclude_pipes:
                    pipe_data.diameter = min(discrete_diameters)
            for pipe, pipe_data in wn.pipes():
                if pipe_data.name in exclude_pipes:
                    pipe_data.diameter = max(discrete_diameters)

            # Add a pump to the network to reduce the neccessity for a larger pipe diameters
            wn.add_pump('PUMP_1',
                        start_node_name='2', 
                        end_node_name='3', 
                        pump_type='POWER',
                        pump_parameter=50.0)

        visualise_demands(wn, title = f'Initial_{name}_network', save_path=os.path.join(script, 'Plots', f'{name}_Initial_Network.png'), show = True)

        # 5. Run simulation on the fully modified model object
        results = run_epanet_simulation(wn)
        
        # 6. Evaluate and print performance metrics
        performance_metrics = evaluate_network_performance(wn, results)

        # print(f"Modified energy consumption for {file}: {energy_consumption:.2f} kWh")
        # print(f"Units of the network: {units}")

        # 7. Write the final, fully modified model to the file
        # print(f"Saving modified network to: {save_path}")
        write_inpfile(wn, save_path)