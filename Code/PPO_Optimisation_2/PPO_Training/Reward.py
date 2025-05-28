
"""

In this file we take the hydraulic simulation results from a network and other global features to determine the reward at a particular time step

"""

# Import libraries

import numpy as np
import wntr
import os
import random
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from wntr.graphics import plot_network
from Visualise_network import visualise_network, visualise_demands

def calculate_reward(
        initial_state, 
        actions, # List of pipe ID diameter pairs representing the actions taken in the current time step
        pipes,  # List of pipes in the network with unit costs
        performance_metrics,
        labour_cost):

    """
    Calculate the reward based on the performance metrics and actions taken

    Args:
        initial_state (dict): The initial state of the network before actions were taken
        actions (list): The actions taken in the current time step. this will be in the form of
        performance_metrics (dict): The performance metrics from the hydraulic simulation
    """

    # wn = wntr.network.WaterNetworkModel(initial_state)

    initial_pipes = list(initial_state.pipes())

    print(f"Length of initial pipes: {len(list(initial_pipes))}")

    energy_cost = performance_metrics['total_pump_cost']

    cost = compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes)
    pressure_deficit = performance_metrics['total_pressure_deficit']
    demand_satisfaction = performance_metrics['demand_satisfaction_ratio']
    disconnection, actions_causing_disconnections = calc_disconnect(initial_state, actions)

    # Create a set of action to describe upgrading all pipes to the maximum diameter
    # Extract diameter data from pipes disctionary
    max_diameter = max([pipes[pipe]['diameter'] for pipe in pipes])
    next_largest = max([pipes[pipe]['diameter'] for pipe in pipes if pipes[pipe]['diameter'] < max_diameter])

    print(f"Max diameter: {max_diameter}, Next largest diameter: {next_largest}")

    # Extract pipe IDs from initial_pipes
    initial_pipe_ids = [pipe_data.name for pipe, pipe_data in initial_pipes]
    max_actions = [(pipe_id, max_diameter) for pipe_id in initial_pipe_ids]

    # Create a new list for corrected max actions
    corrected_max_actions = []
    
    # Check if pipes already have the maximum diameter and adjust accordingly
    for i, (pipe_id, new_diameter) in enumerate(max_actions):
        # Find the current diameter of this pipe
        current_diameter = None
        for pipe, pipe_data in initial_pipes:
            if pipe_data.name == pipe_id:
                current_diameter = pipe_data.diameter
                break
        
        # If the pipe already has the maximum diameter, use next largest instead
        if current_diameter == max_diameter:
            corrected_max_actions.append((pipe_id, next_largest))
        else:
            corrected_max_actions.append((pipe_id, max_diameter))
    
    max_actions = corrected_max_actions
    print(f"Max actions: {max_actions}")

    max_cost = compute_total_cost(initial_pipes, max_actions, labour_cost, energy_cost, pipes)
    cost_ratio = cost / max_cost if max_cost > 0 else 0

    print(f"Cost: {cost}, Max Cost: {max_cost}, Cost Ratio: {cost_ratio}")

    return None

def compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes):
    # pipes in the form {Pipe_ID: {'diameter': diameter, 'unit_cost': unit_cost}}
     
    num_changes = 0
    pipe_upg_cost = 0
    labour_cost_total = 0

    # Create a pipe_id to pipe object mapping
    pipe_dict = {}
    pipe_ids = []
    
    # Convert initial_pipes generator to a usable dictionary
    for pipe_id, pipe_data in initial_pipes:
        pipe_dict[pipe_id] = pipe_data
        pipe_ids.append(pipe_id)

    # Print length of initial pipes
    print(f"Initial pipes length: {len(pipe_dict)}")
        
    # Process each action
    for action in actions:
        pipe_id, new_diameter = action
        if pipe_id in pipe_ids:
            # Access pipe object from our dictionary
            pipe_obj = pipe_dict[pipe_id]
            if pipe_obj.diameter != new_diameter:
                num_changes += 1
                length_of_pipe = pipe_obj.length
                
                # Find the right pipe diameter in pipes dict
                for pipe_type, pipe_data in pipes.items():
                    if pipe_data['diameter'] == new_diameter:
                        pipe_upg_cost += pipe_data['unit_cost'] * length_of_pipe
                        labour_cost_total += labour_cost * length_of_pipe
                        break

    # Add all costs together
    total_cost = pipe_upg_cost + labour_cost_total + energy_cost

    return total_cost

def calc_disconnect(initial_state, actions):
    """
    Calculate the disconnection penalty based on the initial state and actions taken
    """
    # initial_state = wntr.network.WaterNetworkModel(initial_state)
    G = initial_state.to_graph()
    # Convert to undirected graph for analysis of disconnections
    G = G.to_undirected()
    initial_pipes = initial_state.pipes()
    initial_pipe_IDs = [pipe_data.name for pipe, pipe_data in initial_pipes]

    disconnections = False
    actions_casuing_disconnections = []

    # Display graph edges from G
    print(f"Initial graph edges: {list(G.edges)}")

    for action in actions:
        pipe_id, new_diameter = action
        if pipe_id in initial_pipe_IDs:
            # Remove the pipe from the graph
            start_node = initial_state.get_link(pipe_id).start_node
            end_node = initial_state.get_link(pipe_id).end_node
            G.remove_edge(f"{start_node}", f"{end_node}", pipe_id)
            # Check for disconnections
            if not nx.is_connected(G):
                # print(f"Disconnection detected after removing pipe {pipe_id} with new diameter {new_diameter}")
                disconnections = True
                actions_casuing_disconnections.append(action)
            # Readd the pipe
            G.add_edge(f"{start_node}", f"{end_node}", pipe_id)

    return disconnections, actions_casuing_disconnections

if __name__ == "__main__":
    # Example usage
    
    # Create example wntr network
    wn = wntr.network.WaterNetworkModel()
    wn.add_junction('J1', base_demand=0.100, coordinates = (10, 10))
    wn.add_junction('J2', base_demand=0.150, coordinates = (20, 20))
    wn.add_junction('J3', base_demand=0.200, coordinates = (30, 30))
    wn.add_junction('J4', base_demand=0.100, coordinates = (40, 40))
    wn.add_junction('J5', base_demand=0.150, coordinates = (50, 30))
    wn.add_junction('J6', base_demand=0.200, coordinates = (60, 20))
    wn.add_junction('J7', base_demand=0.100, coordinates = (70, 10))
    wn.add_junction('J8', base_demand=0.150, coordinates = (80, 20))

    wn.add_pump('Pump1', 'J1', 'J2', "POWER", 50) # Arbitrary 50kWh pump
    wn.add_reservoir('R1', base_head=50, coordinates = (40, 10))

    pipes = {'Pipe_1': {'diameter': 0.3, 'unit_cost': 100},
            'Pipe_2': {'diameter': 0.4, 'unit_cost': 150},
            'Pipe_3': {'diameter': 0.5, 'unit_cost': 200},
            'Pipe_4': {'diameter': 0.6, 'unit_cost': 250},
            'Pipe_5': {'diameter': 0.7, 'unit_cost': 300}}
    
    # Extract diameters from pipes
    pipe_diameters = [pipe['diameter'] for pipe in pipes.values()]
    
    # Connect the junctions with pipes of different diameters
    wn.add_pipe('P1', 'J1', 'J2', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P2', 'J2', 'J3', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P3', 'J3', 'J4', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P4', 'J4', 'J5', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P5', 'J5', 'J6', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P6', 'J6', 'R1', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P7', 'J1', 'R1', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P8', 'J2', 'J6', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P9', 'J6', 'J7', diameter = random.choice(pipe_diameters), length=100, roughness=100)
    wn.add_pipe('P10', 'J7', 'J8', diameter = random.choice(pipe_diameters), length=100, roughness=100) # This pipe is only connected to the rest of the network by pipe 9, so if pipe 9 is removed this should cause a disconnection

    # Initiali visulaisation
    # plot_network(wn, title="Initial Network", node_size=100)
    # plt.show()
    # visualise_demands(wn, title="Initial Network Demands", show = True)

    # define actions as a list of tuples (pipe_id, new_diameter)
    actions = []
    for pipe in wn.pipes():
        pipe_id = pipe[0]
        new_diameter = random.choice(pipe_diameters)
        actions.append((pipe_id, new_diameter))

    print(f"Actions taken: {actions}")

    initial_pipes = wn.pipes()
    labour_cost = 100 # Arbitrary labour cost per pipe

    results = run_epanet_simulation(wn)
    performance_metrics = evaluate_network_performance(wn, results)

    # # Calculate the total cost of the actions taken
    # total_cost = compute_total_cost(initial_pipes, actions, labour_cost, labour_cost, pipes)  # Assuming energy cost is 50 for this example
    # print(f"Total cost of actions taken: {total_cost}")

    # # Calculate disconnections
    # disconnections, actions_causing_disconnections = calc_disconnect(wn, actions)
    # print(f"Disconnections occurred: {disconnections}")
    # if disconnections:
    #     print(f"Actions causing disconnections: {actions_causing_disconnections}")
    # else:
    #     print("No disconnections occurred.")

    # calculate reward function as a total
    calculate_reward(
        initial_state=wn,
        actions=actions,
        pipes=pipes,
        performance_metrics=performance_metrics,
        labour_cost=labour_cost
    )

    


