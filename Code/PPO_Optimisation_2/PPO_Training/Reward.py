
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
import time

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from wntr.graphics import plot_network
# from Visualise_network import visualise_network, visualise_demands

random.seed(1)

def calculate_reward(
        current_network, 
        original_pipe_diameters,  # Dictionary of original pipe diameters
        actions,                  # List of pipe ID diameter pairs representing the actions
        pipes,                    # Dictionary of pipe types with unit costs
        performance_metrics,
        labour_cost,
        downgraded_pipes,
        disconnections=False,
        actions_causing_disconnections=None,
        max_pd = None,
        max_cost = None):

    """
    Calculate the reward based on the performance metrics and actions taken

    Args:
        current_network: The current state of the network after actions were taken
        original_pipe_diameters: Dictionary of original pipe diameters before actions
        actions: The actions taken in the current time step
        pipes: Dictionary of pipe types with unit costs
        performance_metrics: The performance metrics from the hydraulic simulation
        labour_cost: Cost of labor per meter of pipe
        disconnections: Boolean indicating if actions caused disconnections
        actions_causing_disconnections: List of actions that caused disconnections
    """

    print("Calculating cost given provided actions...")

    reward_weights = [0.3, # Cost ratio
                      0.4, # Pressure deficit ratio
                      0.3, # Demand satisfaction ratio
                      ]

    initial_pipes = list(current_network.pipes())
    energy_cost = performance_metrics['total_pump_cost']
    total_pressure = performance_metrics['total_pressure']

    # Compute cost using original pipe diameters instead of a separate network
    cost = compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes, original_pipe_diameters)
    pressure_deficit = performance_metrics['total_pressure_deficit']
    demand_satisfaction = performance_metrics['demand_satisfaction_ratio']

    # ------------------------------------
    # Extract cost ratio

    # Create a set of action to describe upgrading all pipes to the maximum diameter
    # max_diameter = max([pipes[pipe]['diameter'] for pipe in pipes])
    # next_largest = max([pipes[pipe]['diameter'] for pipe in pipes if pipes[pipe]['diameter'] < max_diameter])

    # print(f"Max diameter: {max_diameter}, Next largest diameter: {next_largest}")

    # Extract pipe IDs from initial_pipes
    # initial_pipe_ids = [pipe_data.name for pipe, pipe_data in initial_pipes]
    # max_actions = [(pipe_id, max_diameter) for pipe_id in initial_pipe_ids]

    # Create a new list for corrected max actions
    # corrected_max_actions = []
    
    # # Check if pipes already have the maximum diameter and adjust accordingly
    # for i, (pipe_id, new_diameter) in enumerate(max_actions):
    #     # Get the current diameter from original_pipe_diameters if available
    #     if pipe_id in original_pipe_diameters:
    #         current_diameter = original_pipe_diameters[pipe_id]
    #     else:
    #         # Otherwise get it from the current network
    #         for pipe, pipe_data in initial_pipes:
    #             if pipe_data.name == pipe_id:
    #                 current_diameter = pipe_data.diameter
    #                 break
        
    #     # If the pipe already has the maximum diameter, use next largest instead
    #     if current_diameter == max_diameter:
    #         corrected_max_actions.append((pipe_id, next_largest))
    #     else:
    #         corrected_max_actions.append((pipe_id, max_diameter))
    
    # max_actions = corrected_max_actions

    # print("-------------------------------------")
    # print("Calculating cost given maximum actions...")

    # max_cost = compute_total_cost(initial_pipes, max_actions, labour_cost, energy_cost, pipes, original_pipe_diameters)

    print("-------------------------------------")
    cost_ratio = max(1 - (cost / max_cost), 0) if max_cost > 0 else 0 # Where a cost of 1 is the best possible outcome
    # ------------------------------------
    
    # Normalise the pressure deficit ratio between 0 and the maximum pressure deficit (taken when all the pipes have the smallest possible diameter from selection)

    pd_ratio = max(1 - (pressure_deficit / max_pd), 0) if max_pd else 0  # Where a PD ratio of 1 is the best possible outcome

    # ------------------------------------
    # Disconnection penalty
    disconnection_multiplier = 0 if disconnections else 1

    # ------------------------------------
    # Demand satisfaction ratio is already between 0 and 1, so we can use it directly
    # print(f"Demand Satisfaction Ratio: {demand_satisfaction}")

    demand_satisfaction = max(demand_satisfaction, 0)  # Ensure it's non-negative
    
    # ------------------------------------
    # Calculate the final reward
    reward = max(reward_weights[0] * cost_ratio +
              reward_weights[1] * pd_ratio +
              reward_weights[2] * demand_satisfaction,
              0)  # Ensure reward is non-negative
    
    """Relaxing this because action mask should prevent any downgrades"""
    # if downgraded_pipes:
    #     reward = 0 # Overwrite reward if any pipes were downgraded to smaller pipes
    
    print(f"Reward: {reward} (Cost Ratio: {cost_ratio}, PD Ratio: {pd_ratio}, Demand Satisfaction: {demand_satisfaction}, Disconnection Multiplier: {disconnection_multiplier}")
    
    print("------------------------------------")

    return reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, downgraded_pipes

def reward_just_pd(
    current_network, 
        original_pipe_diameters,  # Dictionary of original pipe diameters
        actions,                  # List of pipe ID diameter pairs representing the actions
        pipes,                    # Dictionary of pipe types with unit costs
        performance_metrics,
        labour_cost,
        downgraded_pipes,
        disconnections=False,
        actions_causing_disconnections=None,
        max_pd = None,
        max_cost = None):
    
    """This calculates the reward solely as a funciton of the pressure deficit"""

     # Existing calculations can remain if you want to log these values via the info dict
    # The print statement for "Calculating cost given provided actions..." can also remain or be removed.
    # print("Calculating cost given provided actions...") 

    initial_pipes = list(current_network.pipes())
    energy_cost = performance_metrics['total_pump_cost']
    
    # Cost is still computed as it's part of the return tuple and might be logged
    cost = compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes, original_pipe_diameters)
    
    pressure_deficit = performance_metrics['total_pressure_deficit']
    demand_satisfaction = performance_metrics['demand_satisfaction_ratio'] # Still computed for info

    # Original ratio calculations (can be kept for info, but won't be used in the primary reward)
    cost_ratio_info = max(1 - (cost / max_cost), 0) if max_cost is not None and max_cost > 0 else 0
    pd_ratio_info = max(1 - (pressure_deficit / max_pd), 0) if max_pd is not None and max_pd > 0 else 0
    demand_satisfaction_info = max(demand_satisfaction, 0)

    # --- MODIFIED REWARD CALCULATION ---
    # The agent maximizes reward. To minimize pressure deficit, the reward should be -pressure_deficit.
    # A smaller (closer to zero) pressure deficit results in a less negative (i.e., higher) reward.
    reward = -pressure_deficit
    # --- END OF MODIFIED REWARD CALCULATION ---

    # Modify the print statement to reflect the new reward focus
    print("-------------------------------------")
    print(f"Simplified Reward (solely minimizing PD): {reward:.4f} (based on raw Pressure Deficit: {pressure_deficit:.4f})")
    # You can still print other metrics if desired for debugging:
    # print(f"  (For info: Cost={cost:.2f}, Demand Satisfaction={demand_satisfaction_info:.4f})")
    print("------------------------------------")

    # The function signature requires returning all these values.
    # The agent will optimize based on the first 'reward' value.
    return reward, cost, pd_ratio_info, demand_satisfaction_info, disconnections, actions_causing_disconnections, downgraded_pipes

# Add these new functions to your existing Reward.py file

def reward_minimise_pd(performance_metrics: dict, max_pd: float, **kwargs) -> tuple:
    """
    CURRICULUM STAGE 1: Reward function focused solely on minimizing pressure deficit.
    The reward is the negative of the pressure deficit, directly incentivizing the agent
    to reduce it towards zero.
    """
    pressure_deficit = performance_metrics.get('total_pressure_deficit', 0.0)
    
    # Return values for logging purposes, even if not used in reward calculation
    cost = kwargs.get('cost', 0)
    demand_satisfaction = performance_metrics.get('demand_satisfaction_ratio', 0)
    pd_ratio = 1 - (pressure_deficit / max_pd) if max_pd > 0 else 0

    reward = pd_ratio

    return reward, cost, pd_ratio, demand_satisfaction

def reward_pd_and_cost(performance_metrics: dict, cost: float, max_pd: float, max_cost: float, **kwargs) -> tuple:
    """
    CURRICULUM STAGE 2: Reward function that balances minimizing pressure deficit and cost.
    It uses normalized ratios to prevent one objective from dominating the other.
    """
    pressure_deficit = performance_metrics.get('total_pressure_deficit', 0.0)
    
    # Normalize pressure deficit and cost to be between 0 and 1 (where 1 is best)
    pd_ratio = 1 - (pressure_deficit / max_pd) if max_pd > 0 else 0
    cost_ratio = 1 - (cost / max_cost) if max_cost > 0 else 0
    
    # Weights for combining the objectives. Pressure deficit is weighted higher
    # to ensure the agent doesn't forget the lessons from Stage 1.
    w_pd = 0.7
    w_cost = 0.3
    
    reward = (w_pd * pd_ratio) + (w_cost * cost_ratio)

    # Return values for logging
    demand_satisfaction = performance_metrics.get('demand_satisfaction_ratio', 0)

    return reward, cost, pd_ratio, demand_satisfaction

def reward_full_objective(performance_metrics: dict, cost: float, max_pd: float, max_cost: float, **kwargs) -> tuple:
    """
    CURRICULUM STAGE 3: The full, multi-objective reward function from your original code.
    This balances cost, pressure deficit, and demand satisfaction.
    """
    pressure_deficit = performance_metrics.get('total_pressure_deficit', 0.0)
    demand_satisfaction = performance_metrics.get('demand_satisfaction_ratio', 0)
    
    # Normalize metrics
    pd_ratio = 1 - (pressure_deficit / max_pd) if max_pd > 0 else 0
    cost_ratio = 1 - (cost / max_cost) if max_cost > 0 else 0

    # These weights are from your original `calculate_reward` function.
    reward_weights = [0.3, 0.4, 0.2] # Cost, PD, Demand
    
    reward = max(reward_weights[0] * cost_ratio) + \
             (reward_weights[1] * pd_ratio) + \
             (reward_weights[2] * demand_satisfaction, 0) # Ensure reward is non-negative

    return reward, cost, pd_ratio, demand_satisfaction

# Note: The original `calculate_reward` can be kept for comparison or as the final stage reward.
# The disconnection penalty is best handled within the environment's step function, as it's a critical failure state.


def compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes, original_pipe_diameters=None):
    """
    Compute the total cost of actions taken
    
    Args:
        initial_pipes: List of pipes in the current network
        actions: List of pipe ID diameter pairs representing the actions taken
        labour_cost: Cost of labor per meter of pipe
        energy_cost: Energy cost from pump operation
        pipes: Dictionary of pipe types with unit costs
        original_pipe_diameters: Dictionary of original pipe diameters before actions
    """
     
    num_changes = len(actions)
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
            
            # Determine original diameter (either from stored original diameters or current pipe)
            original_diameter = original_pipe_diameters.get(pipe_id, pipe_obj.diameter) if original_pipe_diameters else pipe_obj.diameter

            # print(f"Processing action {action} for pipe {pipe_id}: Original Diameter = {original_diameter}, New Diameter = {new_diameter}")
            
            if new_diameter != original_diameter:
                # num_changes += 1
                length_of_pipe = pipe_obj.length
                
                # Find the right pipe diameter in pipes dict
                for pipe_type, pipe_data in pipes.items():
                    if pipe_data['diameter'] == new_diameter:
                        pipe_upg_cost += pipe_data['unit_cost'] * length_of_pipe
                        labour_cost_total += labour_cost * length_of_pipe
                        break
        else: 
            # New pipes added will have IDs not in the initial pipes
            # For these, we need to estimate a length or get it from the action
            # Assuming a default length if not available
            length_of_pipe = 100  # Default length in meters
            
            for pipe_type, pipe_data in pipes.items():
                if pipe_data['diameter'] == new_diameter:
                    pipe_upg_cost += pipe_data['unit_cost'] * length_of_pipe
                    labour_cost_total += labour_cost * length_of_pipe
                    break

    # Add all costs together
    total_cost = pipe_upg_cost + labour_cost_total + (energy_cost*(365/2)) # Energy in 6 month window

    print(f"Number of changes: {num_changes}, Pipe Upgrade Cost: {pipe_upg_cost}, Labour Cost Total: {labour_cost_total}, Energy Cost: {energy_cost}, Total Cost: {total_cost}")

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
    # print(f"Initial graph edges: {list(G.edges)}")

    for action in actions:
        pipe_id, new_diameter = action
        if pipe_id in initial_pipe_IDs:
            # Remove the pipe from the graph
            start_node = initial_state.get_link(pipe_id).start_node
            end_node = initial_state.get_link(pipe_id).end_node
            G.remove_edge(f"{start_node}", f"{end_node}", pipe_id)
            # Check for disconnections
            if not nx.is_connected(G):
                print(f"Disconnection detected after removing pipe {pipe_id} with new diameter {new_diameter}")
                disconnections = True
                actions_casuing_disconnections.append(action)
            # Readd the pipe
            G.add_edge(f"{start_node}", f"{end_node}", pipe_id)

    return disconnections, actions_casuing_disconnections

def test_reward_random_net():
    
    # Create example wntr network
    wn = wntr.network.WaterNetworkModel()
    wn.add_junction('J1', base_demand=0.100, coordinates = (10, 10), elevation = 10)
    wn.add_junction('J2', base_demand=0.150, coordinates = (20, 20), elevation = 25)
    wn.add_junction('J3', base_demand=0.200, coordinates = (30, 30), elevation = 25)
    wn.add_junction('J4', base_demand=0.100, coordinates = (40, 40), elevation = 15)
    wn.add_junction('J5', base_demand=0.150, coordinates = (50, 30), elevation = 15)
    wn.add_junction('J6', base_demand=0.200, coordinates = (60, 20), elevation = 5)
    wn.add_junction('J7', base_demand=0.100, coordinates = (70, 10), elevation = 0)
    wn.add_junction('J8', base_demand=0.150, coordinates = (80, 20), elevation = 5)

    # wn.add_pump('Pump1', 'J1', 'J2', "POWER", 20) # Arbitrary 50kWh pump
    wn.add_reservoir('R1', base_head=20, coordinates = (40, 10))
    
    """The base head must be greater than the elevation of the highest junction to practically service that junction - with a base head of 30 and a junction with elevation 30, this junction can only be serviced with no roughness in the network at all"""

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
    wn.add_pipe('P11', 'J8', 'J5', diameter = random.choice(pipe_diameters), length=100, roughness=100)

    # Initiali visulaisation
    # plot_network(wn, title="Initial Network", node_size=100)
    # plt.show()
    from Visualise_network import visualise_demands 
    visualise_demands(wn, title="Initial Network Demands", show = True)

    # define actions as a list of tuples (pipe_id, new_diameter)
    actions = []
    for pipe in wn.pipes():
        pipe_id = pipe[0]
        new_diameter = random.choice(pipe_diameters)
        actions.append((pipe_id, new_diameter))

    # print(f"Actions taken: {actions}")

    initial_pipes = wn.pipes()
    labour_cost = 100 # Arbitrary labour cost per pipe (£100/m)

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
    
    reward = calculate_reward(
        initial_state=wn,
        actions=actions,
        pipes=pipes,
        performance_metrics=performance_metrics,
        labour_cost=labour_cost,
    )

    print(f"Reward for the actions taken: {reward}")
    return reward

def test_reward_anytown():

    # Take the anytown network, and generate a random set of actions to take on the action
    inp_file = os.path.join(os.path.dirname(__file__), 'Modified_nets', 'anytown-3.inp')
    wn = wntr.network.WaterNetworkModel(inp_file)

    from Visualise_network import visualise_demands 
    visualise_demands(wn, title="Anytown Network Demands", show = True)

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    pipe_diameters = [pipe['diameter'] for pipe in pipes.values()]
    # Increase the diameter of each pipe in the network to the next highest value in the pipe_diameters list, if already at max leave it as is

    exclude_pipes = ['4', '33', '40', '142', '143']

    actions = []
    for pipe in wn.pipes():
        pipe_id = pipe[0]
        if pipe_id not in exclude_pipes:
            current_diameter = pipe[1].diameter
            # Find the next highest diameter in the pipe_diameters list
            next_diameter = min([d for d in pipe_diameters if d > current_diameter], default=current_diameter)
            actions.append((pipe_id, next_diameter))
        
    # print(f"Original network pipe diameters: {[p[1].diameter for p in wn.pipes()]}")
    # print(f"Actions taken: {actions}")

    # calculate the reward of the actions take
    initial_pipes = wn.pipes()
    labour_cost = 100  # Arbitrary labour cost per pipe (£100/m)
    results = run_epanet_simulation(wn)
    performance_metrics = evaluate_network_performance(wn, results)

    # Identify downgraded pipes
    downgraded_pipes = False
    for action in actions:
        pipe_id, new_diameter = action
        if pipe_id in exclude_pipes:
            continue
        current_diameter = initial_pipes[pipe_id].diameter
        if new_diameter < current_diameter:
            downgraded_pipes = True
            print(f"Pipe {pipe_id} downgraded from {current_diameter} to {new_diameter}")

    # calculate the maximum pressure deficit for the anytown network for when pipes all have the smallest diameter

    min_pipe_diameter = min(pipe_diameters)
    wn_copy = wntr.network.WaterNetworkModel(inp_file)
    for pipe in wn_copy.pipes():
        pipe_id = pipe[0]
        if pipe_id not in exclude_pipes:
            wn_copy.get_link(pipe_id).diameter = min_pipe_diameter
    results_copy = run_epanet_simulation(wn_copy)
    performance_metrics_copy = evaluate_network_performance(wn_copy, results_copy)
    max_pd = performance_metrics_copy['total_pressure_deficit']

    reward = calculate_reward(
        initial_state=wn,
        actions=actions,
        pipes=pipes,
        performance_metrics=performance_metrics,
        labour_cost=labour_cost,
        downgraded_pipes=downgraded_pipes,
        max_pd = max_pd
    )
    print(f"Reward for the actions taken: {reward}")

def plot_diameter_effect_on_reward(inp_file, net_name):
    """
    Test how increasing pipe diameters affects the reward function in the Anytown network.
    Systematically increase diameters by different percentages and plot the results.
    """
    # Load the Anytown network
    script = os.path.dirname(__file__)
    inp_file = os.path.join(os.path.dirname(__file__), 'Modified_nets', inp_file)
    
    # Define pipe data
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    
    pipe_diameters = sorted([pipe['diameter'] for pipe in pipes.values()])
    labour_cost = 100  # Arbitrary labour cost per pipe (£100/m)
    
    # Exclude pipes that shouldn't be modified
    if net_name == 'Anytown Network':
        exclude_pipes = ['4', '33', '40', '142', '143']
    elif net_name == 'Hanoi Network':
        exclude_pipes = ['12', '11', '10', '2', '1', '21', '22']
    
    # Define scenarios for diameter increases
    scenarios = [
        {"name": "Baseline", "increase_level": 0},  # No increase
        {"name": "25% Pipes", "increase_level": 0.25},  # Increase 25% of pipes
        {"name": "50% Pipes", "increase_level": 0.50},  # Increase 50% of pipes
        {"name": "75% Pipes", "increase_level": 0.75},  # Increase 75% of pipes
        {"name": "All Pipes", "increase_level": 1.0}    # Increase all pipes
    ]
    
    # Alternate approach - try increasing to different diameter levels
    diameter_steps = [
        {"name": "Baseline", "step": 0},  # No increase
        {"name": "1 Step Up", "step": 1},  # Increase by 1 diameter step
        {"name": "2 Steps Up", "step": 2},  # Increase by 2 diameter steps
        {"name": "3 Steps Up", "step": 3},  # Increase by 3 diameter steps
        {"name": "4 Steps Up", "step": 4},  # Increase by 4 diameter steps
        {"name": "5 Steps Up", "step": 5},  # Increase by 5 diameter steps
    ]
    
    # Store results
    reward_results = []
    metrics_results = []
    
    # Test each scenario
    for scenario in diameter_steps:
        print(f"\nTesting scenario: {scenario['name']}")
        
        # Load a fresh copy of the network for each scenario
        wn = wntr.network.WaterNetworkModel(inp_file)
        wn_original = wntr.network.WaterNetworkModel(inp_file)  # Keep original for reference

        min_pipe_diameter = min(pipe_diameters)
        wn_copy = wntr.network.WaterNetworkModel(inp_file)
        for pipe in wn_copy.pipes():
            pipe_id = pipe[0]
            if pipe_id not in exclude_pipes:
                wn_copy.get_link(pipe_id).diameter = min_pipe_diameter
        results_copy = run_epanet_simulation(wn_copy)
        performance_metrics_copy = evaluate_network_performance(wn_copy, results_copy)
        max_pd = performance_metrics_copy['total_pressure_deficit']
        
        # Generate actions based on the scenario
        actions = []
        all_pipes = list(wn.pipes())
        
        # If it's a step-based scenario
        step = scenario['step']
        
        for pipe in all_pipes:
            pipe_id = pipe[0]
            
            if pipe_id not in exclude_pipes:
                current_diameter = pipe[1].diameter
                current_index = -1
                
                # Find the current index in our diameter options
                for i, d in enumerate(pipe_diameters):
                    if abs(current_diameter - d) < 0.0001:  # Use small epsilon for float comparison
                        current_index = i
                        break
                
                # If current diameter isn't in our list, find the closest one
                # if current_index == -1:
                #     current_index = min(range(len(pipe_diameters)), 
                #                       key=lambda i: abs(pipe_diameters[i] - current_diameter))
                
                # Calculate new diameter based on step
                if step == 0:  # Baseline - no change
                    new_diameter = current_diameter
                elif step == 999:  # Max diameter
                    new_diameter = pipe_diameters[-1]
                else:
                    # Increase by step, but don't exceed max diameter
                    new_index = min(current_index + step, len(pipe_diameters) - 1)
                    new_diameter = pipe_diameters[new_index]
                
                # Only add an action if the diameter changes
                # if abs(new_diameter - current_diameter) > 0.0001:
                actions.append((pipe_id, new_diameter))

        # Apply actions to the network
        for pipe_id, new_diameter in actions:
            if pipe_id in wn.pipe_name_list:
                wn.get_link(pipe_id).diameter = new_diameter

        # Run simulation and calculate reward
        results = run_epanet_simulation(wn)
        performance_metrics = evaluate_network_performance(wn, results)
        
        # Calculate the reward
        reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, downgraded_pipes = calculate_reward(
            current_network=wn,
            original_pipe_diameters={pipe[0]: pipe[1].diameter for pipe in wn_original.pipes()},
            actions=actions,
            pipes=pipes,
            performance_metrics=performance_metrics,
            labour_cost=labour_cost,
            downgraded_pipes=False,  # Assume no downgraded pipes for this test
            max_pd=max_pd
        )
        
        # Store results
        reward_results.append({
            "scenario": scenario['name'],
            "reward": reward,
            "actions_count": len(actions)
        })
        
        # Store key metrics
        metrics_results.append({
            "scenario": scenario['name'],
            "total_cost": cost,
            "pressure deficit ratio": pd_ratio,
            "demand satisfaction ratio": demand_satisfaction,
        })
        
        print(f"Scenario: {scenario['name']}, Actions: {len(actions)}, Reward: {reward}")
    
    # Create a DataFrame for plotting
    df_reward = pd.DataFrame(reward_results)
    df_metrics = pd.DataFrame(metrics_results)
    
    # Plot the rewards
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_reward['scenario'], df_reward['reward'], color='steelblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.title(f'Effect of Pipe Diameter Increases on Network Reward for {net_name}')
    plt.xlabel('Scenario')
    plt.ylabel('Reward Value')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join('Plots', 'Tests', f'diameter_effect_on_reward_{net_name}.png'))
    plt.show()
    
    # Plot key metrics
    fig = plt.figure(figsize=(16, 5))
    
    metrics = ['total_cost', 'pressure deficit ratio', 'demand satisfaction ratio']
    titles = [f'Total Cost of Actions for {net_name}',
              f'Pressure Deficit Ratio for {net_name}',
              f'Demand Satisfaction Ratio for {net_name}']
    
    # Create a 2x2 grid but only use 3 positions
    grid_positions = [(0, 0), (0, 1), (0, 2)]  
    
    for i, ((row, col), metric, title) in enumerate(zip(grid_positions, metrics, titles)):
        # Create subplot at the specified position
        ax = plt.subplot2grid((1, 3), (row, col))
        
        # Plot the bars
        if metric == 'total_cost':
            # Convert cost to millions for better readability
            cost_in_millions = df_metrics[metric] / 1e6
            bars = ax.bar(df_metrics['scenario'], cost_in_millions)
            
            # Add value labels in millions format
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'£{height:.2f}M', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel('Total Cost (£ Millions)')
        else:
            bars = ax.bar(df_metrics['scenario'], df_metrics[metric])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel(metric.replace('_', ' ').title())

        ax.set_title(title, fontsize = 10)
        ax.set_xlabel('Scenario', fontsize = 8)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(axis='y', linestyle='--')
        
        # Rotate x-axis labels if they're too long
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join('Plots', 'Tests', f'diameter_effect_on_metrics_{net_name}.png'))
    plt.show()
    
    return df_reward, df_metrics

if __name__ == "__main__":
    # test_reward_random_net()
    # test_reward_random_net()
    # print("Test completed successfully.")
    # test_reward_anytown(

    # Visualise both networks
    script = os.path.dirname(__file__)
    inp_file_anytown = os.path.join(script, 'Modified_nets', 'anytown_sprawling_3', 'Step_50.inp')
    inp_file_hanoi = os.path.join(script, 'Modified_nets', 'hanoi_sprawling_3', 'Step_50.inp')
    wn_anytown = wntr.network.WaterNetworkModel(inp_file_anytown)
    wn_hanoi = wntr.network.WaterNetworkModel(inp_file_hanoi)
    # visualise_demands(wn_anytown, title="Anytown Network Demands", show = True)
    # visualise_demands(wn_hanoi, title="Hanoi Network Demands", show = True)

    # Print existing pipe diameters
    print("Anytown Network Pipe Diameters:")
    for pipe in wn_anytown.pipes():
        print(f"{pipe[0]}: {pipe[1].diameter} m")
    print("Hanoi Network Pipe Diameters:")
    for pipe in wn_hanoi.pipes():
        print(f"{pipe[0]}: {pipe[1].diameter} m")
    

    print("--------------------------------------------------")
    print("Testing diameter effect on reward for Anytown and Hanoi networks...")
    print("--------------------------------------------------")
    print("Anytown Network Pipe Diameters:")
    df_rewards_anytown, df_metrics_anytown = plot_diameter_effect_on_reward('anytown-3.inp', 'Anytown Network')
    print("--------------------------------------------------")
    print("Hanoi Network Pipe Diameters:")
    df_rewards_hanoi, df_metrics_hanoi = plot_diameter_effect_on_reward('hanoi-3.inp', 'Hanoi Network')

