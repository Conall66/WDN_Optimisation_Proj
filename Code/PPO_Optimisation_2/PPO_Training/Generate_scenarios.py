
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
def generate_final_sprawl_net(wn, sprawl_percentage=0.05, min_dist = 1000, max_dist = 3000):
    """
    This function generates a final sprawling network by adding 5% more nodes to the existing network.
    The new nodes are added at random positions and connected to the nearest existing node.
    """

    # wn = wntr.network.WaterNetworkModel(wn)

    nodes_to_add = int(len(wn.nodes) * sprawl_percentage)
    # fringe_nodes = identify_fringe_nodes(wn)

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
        
        while True:
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
                break
            else:
                # Remove the temporary node if it is not suitable
                wn.remove_node(new_node_id)

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
    return wn

# Given sprawling network, generate a final network with 5% more nodes

# Given start and end networks, generate step by step tranisition networks for 6 month gap following transition state rules

# Given network, allocate demands (showing consideration for current distribution)

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
    
    print("Scenario folders created successfully.")

    for scenario in scenarios:
        scenario_path = os.path.join(base_path, scenario)
        print(f"Processing scenario: {scenario}")
        
        # Load the initial network based on the scenario
        if 'anytown' in scenario:
            inp_file = os.path.join(script, 'Modified_nets', 'anytown-3.inp')
        else:
            inp_file = os.path.join(script, 'Modified_nets', 'hanoi-3.inp')
        
        wn = wntr.network.WaterNetworkModel(inp_file)
        
        # Generate a sprawling network and transitionary states
        if scenario.endswith('sprawling_1') or scenario.endswith('sprawling_2') or scenario.endswith('sprawling_3'):

            wn = generate_final_sprawl_net(wn, 0.1)
            # Visualise the sprawling network
            plot_network(wn, title=f"Sprawling Network: {scenario}")
            # Store the sprawlin networks in their scenario folder
            