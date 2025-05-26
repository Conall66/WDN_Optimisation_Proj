
"""

In this script, we take the initial water distribution networks as inputs and for each time step of 6 months, and for each scenario combination, generate a set of future networks for the PPO agent to pool from.

"""

# Import libraries

import os
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wntr.network.io import write_inpfile
from wntr.graphics.network import plot_network
import random
from copy import deepcopy

from Demand_gen import generate_demand_curves
from Visualise_network import visualise_network, visualise_demands

# Generate subfolders for each scenario combination in Plots folder
"""
anytown or hanoi:
    sprawling or densifying:
        demand_growth_rate:

(budget will not be encaptured here but chosen at random by the agent at the start of each episode)
"""
def generate_scenario_folders(base_path, scenarios):
    for scenario in scenarios:
        scenario_path = os.path.join(base_path, scenario)
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)
            print(f"Created folder: {scenario_path}")

def calc_distance(node1_id, node2_id, wn):
    """
    Calculate the Euclidean distance between two nodes.
    """
    node1 = wn.get_node(node1_id)
    node2 = wn.get_node(node2_id)

    # print(f"Node coordinates of {node1_id}: {node1.coordinates}")
    # print(f"Node coordinates of {node2_id}: {node2.coordinates}")

    x1, y1 = node1.coordinates
    x2, y2 = node2.coordinates

    # x1, y1 = node1_id.cordinates
    # x2, y2 = node2_id.coordinates
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def identify_fringe_nodes(wn):
    # For each node, determine their position from the reservoirs and tanks - sort so that furthese nodes are at the top of the list

    # wn = wntr.network.WaterNetworkModel(wn)

    fringe_nodes = []
    for node, node_data in wn.junctions():
        # Calculate distance to all reservoirs and tanks
        distances = []
        for reservoir, res_data in wn.reservoirs():
            node_id = node_data.name
            reservoir_id = res_data.name
            distances.append(calc_distance(node_id, reservoir_id, wn))
        for tank, tank_data in wn.tanks():
            node_id = node_data.name
            tank_id = tank_data.name
            distances.append(calc_distance(node_id, tank_id, wn))
        fringe_nodes.append((node, min(distances)))
    # Sort fringe nodes by distance to the nearest reservoir or tank
    fringe_nodes.sort(key=lambda x: x[1], reverse=True)
    # Return only the nodes, not the distances
    return [node for node, _ in fringe_nodes]

# For each network, determine a set of candidate sprawl positions and connections
def generate_final_sprawl_net(wn, net, sprawl_percentage=0.05, min_dist = 750, max_dist = 2000):
    """
    This function generates a final sprawling network by adding 5% more nodes to the existing network.
    The new nodes are added at random positions and connected to the nearest existing node.
    """

    # wn = wntr.network.WaterNetworkModel(wn)

    nodes_to_add = int(len(wn.nodes) * sprawl_percentage)
    # fringe_nodes = identify_fringe_nodes(wn)

    if net == 'anytown':
        for i in range(nodes_to_add):

            # Identify area of existing wn
            max_x = max(wn.nodes[node_id].coordinates[0] for node_id in wn.junction_name_list)
            max_y = max(wn.nodes[node_id].coordinates[1] for node_id in wn.junction_name_list)
            min_x = min(wn.nodes[node_id].coordinates[0] for node_id in wn.junction_name_list)
            min_y = min(wn.nodes[node_id].coordinates[1] for node_id in wn.junction_name_list)
            fringe_nodes = identify_fringe_nodes(wn)

            print(f"Fringe nodes sorted by distance: {[node for node in fringe_nodes]}")

            # fringe_node = random.choice(fringe_nodes)
            new_node_id = f"{len(wn.nodes) + i + 1}"
        
            attempt_count = 0
            attempts = 100
            while attempt_count < attempts:
                # Generate random position along the edges of the existing network
                wall = random.choice(['top', 'bottom', 'left', 'right'])
                if wall == 'top':
                    x = int(random.uniform(min_x, max_x))
                    y = int(max_y + random.uniform(min_dist, max_dist))
                elif wall == 'bottom':
                    x = int(random.uniform(min_x, max_x))
                    y = int(min_y - random.uniform(min_dist, max_dist))
                elif wall == 'left':
                    x = int(min_x - random.uniform(min_dist, max_dist))
                    y = int(random.uniform(min_y, max_y))
                elif wall == 'right':
                    x = int(max_x + random.uniform(min_dist, max_dist))
                    y = int(random.uniform(min_y, max_y))

                """All junctions furthest from the reservoir in the anytown network are ofc at the top of the network. This means that new nodes are tending to be added in the same places every time, which is not ideal."""

                wn.add_junction(new_node_id, elevation=0, base_demand=0, coordinates = (x, y))  # Add a temporary node to check distance

                # Identify the nearest node in the existing network
                nearest_node = None
                nearest_distance = float('inf')
                for node, node_data in wn.junctions():
                    node_id = node_data.name
                    if node_id != new_node_id:
                        distance = calc_distance(new_node_id, node_id, wn)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_node = node

                # print(f"Nearest node to {new_node_id} is {nearest_node} at distance {nearest_distance}")
                fringe_subsset = fringe_nodes[0:int(len(fringe_nodes) * 0.5)]  # Get the first 20% of fringe nodes
                # print(f"Fringe subset: {[node for node in fringe_subsset]}")

                if nearest_node in fringe_subsset and min_dist < nearest_distance < max_dist: # Only add new nodes to the fringe nodes
                    wn.remove_node(new_node_id) # Gets added again later
                    break
                else:
                    # Remove the temporary node if it is not suitable
                    wn.remove_node(new_node_id)
                    attempt_count += 1

            # Determine elevation of the new node based on the nearest node
            nearest_node_id = wn.get_node(nearest_node)

            new_node_elevation = round(nearest_node_id.elevation + random.uniform(-10, 10), 2)
            wn.add_junction(new_node_id, elevation=new_node_elevation, base_demand=0, coordinates=(x, y))  # Add the new node with the calculated elevation

            # Find the next nearest node
            second_nearest_node = None
            second_nearest_distance = float('inf')
            for node, node_data in wn.junctions():
                node_id = node_data.name
                if node_id != nearest_node and node_id != new_node_id:
                    distance = calc_distance(new_node_id, node_id, wn)
                    if distance < second_nearest_distance:
                        second_nearest_distance = distance
                        second_nearest_node = node_id  # Store just the ID

            # Add nearest and second nearest nodes to array
            nearest_nodes = [nearest_node, second_nearest_node]

            print(f"New node {new_node_id} added at coordinates ({x}, {y}) with elevation {new_node_elevation}. Connected to nearest nodes: {nearest_nodes}")

            for nearest_node in nearest_nodes:

                # Get adjoining node id
                connecting_node = wn.get_node(nearest_node)
                connecting_node_id = connecting_node.name

                print(f"Connecting new node {new_node_id} to nearest node {connecting_node_id}")

                # if nearest_nodes[j].name != new_node_id:
                new_pipe_id = f"pipe_{new_node_id}_{connecting_node_id}"

                length = calc_distance(wn.get_node(new_node_id), connecting_node, wn)
                wn.add_pipe(new_pipe_id, new_node_id, connecting_node_id, length = length, diameter=0.0, roughness=0.0, minor_loss = 0) # Start with 0 diameter so agent learns to allocate it

    elif net == 'hanoi':
        # Identify start and end nodes
        start_node = '13'
        end_node = '31' # Junctions manually identified to connect to

        # Identify the difference in x coordinates between the start and end nodes
        start_coordinates = wn.get_node(start_node).coordinates
        end_coordinates = wn.get_node(end_node).coordinates
        x_diff = end_coordinates[0] - start_coordinates[0]
        iterative_dist = x_diff / nodes_to_add

        prev_node = start_node # Node ID
        for node in range(nodes_to_add):
            # Identify start and end nodes
            if node != nodes_to_add - 1:
                # Extract prev node coordinates
                prev_node_coordinates = wn.get_node(prev_node).coordinates
                # Move a random distance to the left from the previous node within acceptable bounds
                new_x = prev_node_coordinates[0] + iterative_dist * random.uniform(1.0, 1.2)
                new_y = prev_node_coordinates[1]
                new_node_id = f"{len(wn.nodes) + node + 1}"
                wn.add_junction(new_node_id, elevation=0, base_demand=0, coordinates=(new_x, new_y))  # Add a temporary node to check distance

                # Add a pipe to the previous node
                new_pipe_id = f"pipe_{new_node_id}_{prev_node}"
                print(f"Adding pipe {new_pipe_id} from {new_node_id} to {prev_node}")
                length = calc_distance(wn.get_node(new_node_id), wn.get_node(prev_node), wn)
                wn.add_pipe(new_pipe_id, new_node_id, prev_node, length=length, diameter=0.0, roughness=0.0, minor_loss=0)  # Start with 0 diameter so agent learns to allocate it

                # Update the previous node to be the new node
                prev_node = new_node_id

            elif node == nodes_to_add -1:
                new_x = end_coordinates[0]
                new_y = start_coordinates[1]
                new_node_id = f"{len(wn.nodes) + node + 1}"
                wn.add_junction(new_node_id, elevation=0, base_demand=0, coordinates=(new_x, new_y))  # Add a temporary node to check distance
                # Add a pipe to the end node
                new_pipe_id = f"pipe_{new_node_id}_{end_node}"
                length = calc_distance(wn.get_node(new_node_id), wn.get_node(end_node), wn)
                wn.add_pipe(new_pipe_id, new_node_id, end_node, length=length, diameter=0.0, roughness=0.0, minor_loss=0)
                # Add a pipe to the previous node
                new_pipe_id = f"pipe_{new_node_id}_{prev_node}"
                length = calc_distance(wn.get_node(new_node_id), wn.get_node(prev_node), wn)
                wn.add_pipe(new_pipe_id, new_node_id, prev_node, length=length, diameter=0.0, roughness=0.0, minor_loss=0)
                
    return wn

# Given start and end networks, determine which update steps add new nodes
def generate_transition_states(start_wn, end_wn, scenario, num_steps = 50, net_save_path = None, plot_save_path = None):
    # Divide the number of nodes difference between the 2 states by the number of steps to determine how many nodes to add at each step - accumulate and when integer value reached new node added. Store these networks in the modified nets folder
    start_nodes = len(start_wn.nodes)
    end_nodes = len(end_wn.nodes)
    node_diff = end_nodes - start_nodes
    nodes_per_step = node_diff / num_steps

    # From both the hanoi and anytown networks, extract the junctions and pipes added in the end_wn
    start_junctions = set(start_wn.junctions())
    end_junctions = set(end_wn.junctions())
    added_junctions = sorted(end_junctions - start_junctions) # This ensure the junctions are sorted by their ID, with the lowest ID first
    
    # For each node in added_junctions, identify the list of pipes connected to it and store
    added_pipes = []
    for junction, junction_data in added_junctions:
        junction_id = junction_data.name
        pipes = start_wn.get_node(junction_id).pipes
        for pipe in pipes:
            pipe_data = start_wn.get_pipe(pipe)
            added_pipes.append((pipe_data.name, pipe_data))
    
    prev_network = start_wn
    add_nodes = 0 # This value updates with each step, and when the value breaches 1 a new node is added and 1 subtracted from the value

    total_initial_demand = 0
    # Update the demands at each junction
    for junction, junction_data in start_wn.junctions():
        total_initial_demand += junction_data.base_demand
        # Set the base demand to a random value between 0 and 10

    # Generate demand curve and extract value for time step
    demand_vals = generate_demand_curves(num_steps, plot=False)

    for step in range(num_steps):
        # Create a new wntr model and add the n
        new_wn = deepcopy(prev_network)  # Create a copy of the previous network
        add_nodes += nodes_per_step  # Increment the number of nodes to add
        while add_nodes >= 1:
            # Add a new junction and pipe from the added junctions and pipes
            if added_junctions:
                new_junction = added_junctions.pop(0)
                new_wn.add_junction(new_junction[0], elevation=new_junction[1].elevation, base_demand=new_junction[1].base_demand, coordinates=new_junction[1].coordinates)
            # Iterate through added_pipes to add the pipes connected to the new junction
            for pipe, pipe_data in added_pipes:
                if pipe_data.start_node == new_junction[0] or pipe_data.end_node == new_junction[0]:
                    new_wn.add_pipe(pipe, pipe_data.start_node, pipe_data.end_node, length=pipe_data.length, diameter=pipe_data.diameter, roughness=pipe_data.roughness, minor_loss=pipe_data.minor_loss)
            add_nodes -= 1  # Subtract 1 from the number of nodes to add

        # Update the demands at each junction
        scenario_index = int(scenario.split('_')[-1]) - 1  # Extract the scenario index from the scenario name
        demand_curve = demand_vals[scenario_index]
        demand_multiplier = demand_curve[step]
        
        # Calculate total initial demand scaled by the multiplier
        target_demand = total_initial_demand * demand_multiplier
        
        # Update demands using the helper function - takes as input the previous network, current network and target demand
        new_wn = update_demands(prev_network, new_wn, target_demand)
          
        # Save the new network to the modified nets folder
        if net_save_path:
            new_file_name = f"Step_{step + 1}.inp"
            new_file_path = os.path.join(net_save_path, new_file_name)
            write_inpfile(new_wn, new_file_path)
            # print(f"Saved network at step {step + 1} to {new_file_path}")

        # Visualise the network and save the plot
        if plot_save_path and step % 10 == 0:  # Save a plot every 10 steps
            plot_file_name = f"Step_{step + 1}.png"
            plot_file_path = os.path.join(plot_save_path, plot_file_name)
            visualise_demands(new_wn, title=f"Step {step + 1} - {scenario}", save_path=plot_file_path, show=False)
            
            # print(f"Saved plot at step {step + 1} to {plot_file_path}")

def update_demands(prev_network, current_network, target_demand):
    """
    Update the demands at each junction in the current network to match the target demand.
    The target demand is calculated based on the total initial demand scaled by a multiplier.
    """
    new_network = deepcopy(current_network)  # Create a copy of the current network to avoid modifying it directly

    # Calculate the total initial demand in the previous network
    total_prev_demand = sum(junction.demand_timeseries_list[0].base_value for _, junction in prev_network.junctions())

    # Calculate how much demand needs to be added
    demand_difference = target_demand - total_prev_demand
    num_junctions = len(list(new_network.junctions()))
    demand_inc = demand_difference / num_junctions

    # For each junction in the current network
    prev_junction_ids = {j.name for _, j in prev_network.junctions()}

    for junction_id, junction in new_network.junctions():
        if junction_id in prev_junction_ids:
            # Existing junction: increase its demand by demand_inc
            prev_junct = prev_network.get_node(junction_id)
            prev_demand = prev_junct.demand_timeseries_list[0].base_value
            junction.demand_timeseries_list[0].base_value = prev_demand + demand_inc
        else:
            # New junction: assign demand_inc as its base demand
            junction.demand_timeseries_list[0].base_value = demand_inc

    return new_network

if __name__ == "__main__":
    # Define the base path for the plots
    script = os.path.dirname(__file__)
    base_path = os.path.join(script, 'Plots', 'Scenarios')
    
    # Define the scenarios
    scenarios = [
        'anytown_sprawling_1',
        'anytown_sprawling_2',
        'anytown_sprawling_3',
        'anytown_densifying_1',
        'anytown_densifying_2',
        'anytown_densifying_3',
        'hanoi_sprawling_1',
        'hanoi_sprawling_2',
        'hanoi_sprawling_3',
        'hanoi_densifying_1',
        'hanoi_densifying_2',
        'hanoi_densifying_3'
    ]
    
    # Generate scenario folders
    generate_scenario_folders(base_path, scenarios)

    new_base_path = os.path.join(script, 'Modified_nets')
    generate_scenario_folders(new_base_path, scenarios)
    
    print("Scenario folders created successfully.")

    for scenario in scenarios:
        plot_scenario_path = os.path.join(base_path, scenario)
        scenario_path = os.path.join(new_base_path, scenario)
        print(f"Processing scenario: {scenario}")
        
        # Load the initial network based on the scenario
        if 'anytown' in scenario:
            inp_file = os.path.join(script, 'Modified_nets', 'anytown-3.inp')
        else:
            inp_file = os.path.join(script, 'Modified_nets', 'hanoi-3.inp')
        
        start_wn = wntr.network.WaterNetworkModel(inp_file)
        
        # Generate a sprawling network and transitionary states
        if scenario.endswith('sprawling_1') or scenario.endswith('sprawling_2') or scenario.endswith('sprawling_3'):

            # Extract whether network is anytown or hanoi
            network = 'anytown' if 'anytown' in scenario.split('_')[0] else 'hanoi'
            final_wn = generate_final_sprawl_net(start_wn, network, 0.1)
            # Visualise the sprawling network
            # plot_network(wn, title=f"Sprawling Network: {scenario}")
            # Store the sprawlin networks in their scenario folder
            generate_transition_states(start_wn, final_wn, scenario, num_steps=50, net_save_path=scenario_path, plot_save_path=plot_scenario_path)

        elif scenario.endswith('densifying_1') or scenario.endswith('densifying_2') or scenario.endswith('densifying_3'):
            # Extract whether network is anytown or hanoi
            network = 'anytown' if 'anytown' in scenario.split('_')[0] else 'hanoi'
            # Visualise the densifying network
            # plot_network(wn, title=f"Densifying Network: {scenario}")
            # Store the densifying networks in their scenario folder
            final_wn = start_wn
            generate_transition_states(start_wn, final_wn, scenario, num_steps=50, net_save_path=scenario_path, plot_save_path=plot_scenario_path)
            