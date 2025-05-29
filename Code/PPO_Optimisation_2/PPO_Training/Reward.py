
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

random.seed(1)

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

    print("Calculating cost given provided actions...")

    # wn = wntr.network.WaterNetworkModel(initial_state)

    reward_weights = [0.1, # Cost ratio
                      0.2, # Pressure deficit ratio
                      0.3, # Demand satisfaction ratio
                      0.4 # Disconnection multiplier
                      ]

    initial_pipes = list(initial_state.pipes())
    # print(f"Length of initial pipes: {len(list(initial_pipes))}")
    energy_cost = performance_metrics['total_pump_cost']
    total_pressure = performance_metrics['total_pressure']

    cost = compute_total_cost(initial_pipes, actions, labour_cost, energy_cost, pipes)
    pressure_deficit = performance_metrics['total_pressure_deficit']
    demand_satisfaction = performance_metrics['demand_satisfaction_ratio']
    disconnection, actions_causing_disconnections = calc_disconnect(initial_state, actions)

    # ------------------------------------
    # Extract cost ratio

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
    # print(f"Max actions: {max_actions}")

    print("-------------------------------------")
    print("Calculating cost given maximum actions...")

    max_cost = compute_total_cost(initial_pipes, max_actions, labour_cost, energy_cost, pipes)
    cost_ratio = 1 - (cost / max_cost) if max_cost > 0 else 0 # Where a cost of 1 is the best possible outcome

    print(f"Cost: {cost}, Max Cost: {max_cost}, Cost Ratio: {cost_ratio}")

    # ------------------------------------
    # Extract pressure deficit ratio from total pressure in the system and pressure deficit
    # pressure deficit ratio is 1 if there is no pressure deficit, and 0 if the pressure deficit is equal to the total pressure in the system

    if pressure_deficit <= 0:
        # No pressure deficit - best case
        pd_ratio = 1
    else:
        # Normalize against total pressure to get ratio between 0 and 1
        pd_ratio = max(0, 1 - (pressure_deficit / total_pressure)) if total_pressure > 0 else 0
    
    print(f"Pressure Deficit: {pressure_deficit}, Total Pressure: {total_pressure}, PD Ratio: {pd_ratio}")

    # ------------------------------------
    # Disconnection penalty
    if disconnection:
        # If there is a disconnection, we apply a penalty
        disconnection_multiplier = 0
    else:
        # If there is no disconnection, we apply a reward
        disconnection_multiplier = 1

    # ------------------------------------
    # Demand satisfaction ratio is already between 0 and 1, so we can use it directly
    print(f"Demand Satisfaction Ratio: {demand_satisfaction}")
    # ------------------------------------
    # Calculate the final reward
    reward = (reward_weights[0] * cost_ratio +
              reward_weights[1] * pd_ratio +
              reward_weights[2] * demand_satisfaction +
              reward_weights[3] * disconnection_multiplier)
    
    print("------------------------------------")

    return reward, cost, pd_ratio, demand_satisfaction, disconnection, actions_causing_disconnections


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
            if abs(pipe_obj.diameter - new_diameter) > 0.001:
                # print(f"Upgrading pipe {pipe_id} from diameter {pipe_obj.diameter} to {new_diameter}")
                num_changes += 1
                length_of_pipe = pipe_obj.length
                
                # Find the right pipe diameter in pipes dict
                for pipe_type, pipe_data in pipes.items():
                    if pipe_data['diameter'] == new_diameter:
                        pipe_upg_cost += pipe_data['unit_cost'] * length_of_pipe
                        labour_cost_total += labour_cost * length_of_pipe
                        break

        else: # New pipes added will have IDs not in the initial pipes
            for pipe_type, pipe_data in pipes.items():
                if pipe_data['diameter'] == new_diameter:
                    pipe_upg_cost += pipe_data['unit_cost'] * pipe_obj.length
                    labour_cost_total += labour_cost * pipe_obj.length
                    break

    # Add all costs together
    total_cost = pipe_upg_cost + labour_cost_total + energy_cost

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
        labour_cost=labour_cost
    )

    print(f"Reward for the actions taken: {reward}")
    return reward

def test_reward_anytown():

    # Take the anytown network, and generate a random set of actions to take on the action
    inp_file = os.path.join(os.path.dirname(__file__), 'Modified_nets', 'anytown-3.inp')
    wn = wntr.network.WaterNetworkModel(inp_file)

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
    reward = calculate_reward(
        initial_state=wn,
        actions=actions,
        pipes=pipes,
        performance_metrics=performance_metrics,
        labour_cost=labour_cost
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
                if abs(new_diameter - current_diameter) > 0.0001:
                    actions.append((pipe_id, new_diameter))

        # Apply actions to the network
        for pipe_id, new_diameter in actions:
            if pipe_id in wn.pipe_name_list:
                wn.get_link(pipe_id).diameter = new_diameter

        # Run simulation and calculate reward
        results = run_epanet_simulation(wn)
        performance_metrics = evaluate_network_performance(wn, results)
        
        reward, cost, pd_ratio, demand_satisfaction, disconnection, actions_causing_disconnections  = calculate_reward(
            initial_state=wn_original,
            actions=actions,
            pipes=pipes,
            performance_metrics=performance_metrics,
            labour_cost=labour_cost
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
    fig = plt.figure(figsize=(12, 10))
    
    metrics = ['total_cost', 'pressure deficit ratio', 'demand satisfaction ratio']
    titles = [f'Total Cost of Actions for {net_name}',
              f'Pressure Deficit Ratio for {net_name}',
              f'Demand Satisfaction Ratio for {net_name}']
    
    # Create a 2x2 grid but only use 3 positions
    grid_positions = [(0, 0), (0, 1), (1, 0)]  
    
    for i, ((row, col), metric, title) in enumerate(zip(grid_positions, metrics, titles)):
        # Create subplot at the specified position
        ax = plt.subplot2grid((2, 2), (row, col))
        
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
    # test_reward_anytown()

    # Visualise both networks
    script = os.path.dirname(__file__)
    inp_file_anytown = os.path.join(script, 'Modified_nets', 'anytown-3.inp')
    inp_file_hanoi = os.path.join(script, 'Modified_nets', 'hanoi-3.inp')
    wn_anytown = wntr.network.WaterNetworkModel(inp_file_anytown)
    wn_hanoi = wntr.network.WaterNetworkModel(inp_file_hanoi)
    visualise_demands(wn_anytown, title="Anytown Network Demands", show = True)
    visualise_demands(wn_hanoi, title="Hanoi Network Demands", show = True)

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

