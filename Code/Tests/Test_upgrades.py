import os
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PPO_Environment2 import WNTRGymEnv

# filepath: test_PPO_Environment2.py

import matplotlib.pyplot as plt

# Configure test parameters
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32}, 
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47}, 
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}

NETWORK_CONFIG = {'max_nodes': 150, 'max_pipes': 200}
BUDGET_CONFIG = {
    "initial_budget_per_step": 500000.0,  # Higher budget for testing all upgrades
    "start_of_episode_budget": 10000000.0,
    "ongoing_debt_penalty_factor": 0.0001,
    "max_debt": 2000000.0,
    "labour_cost_per_meter": 100.0
}
REWARD_CONFIG = {'mode': 'full_objective', 'max_pd_normalization': 5000000.0, 'max_cost_normalization': 10000000.0}
HANOI_SCENARIOS = ['hanoi_sprawling_3']  # Using a single scenario for simplicity
ANYTOWN_SCENARIOS = ['anytown_sprawling_3']  # Added Anytown scenario

@pytest.fixture
def env():
    """Create a test environment with Hanoi network."""
    env = WNTRGymEnv(
        pipes_config=PIPES_CONFIG,
        scenarios=HANOI_SCENARIOS,
        network_config=NETWORK_CONFIG,
        budget_config=BUDGET_CONFIG,
        reward_config=REWARD_CONFIG
    )
    # Reset with specific scenario to ensure consistency
    obs, info = env.reset(options={'scenario_name': HANOI_SCENARIOS[0]})
    yield env
    env.close()

@pytest.fixture
def anytown_env():
    """Create a test environment with Anytown network."""
    env = WNTRGymEnv(
        pipes_config=PIPES_CONFIG,
        scenarios=ANYTOWN_SCENARIOS,
        network_config=NETWORK_CONFIG,
        budget_config=BUDGET_CONFIG,
        reward_config=REWARD_CONFIG
    )
    # Reset with specific scenario to ensure consistency
    obs, info = env.reset(options={'scenario_name': ANYTOWN_SCENARIOS[0]})
    yield env
    env.close()

def test_step_action_application(env):
    """Test that step applies actions correctly to the network."""
    # Get initial pipe diameter for the first pipe
    pipe_name = env.pipe_names[0]
    initial_diameter = env.current_network.get_link(pipe_name).diameter
    
    # Take action to upgrade first pipe (action 1 = first diameter option)
    new_diameter = env.pipe_diameter_options[0]
    obs, reward, terminated, truncated, info = env.step(1)
    
    # Check if diameter didn't change (action 1 should be the smallest diameter)
    # or did change if initial_diameter was smaller
    if initial_diameter < new_diameter:
        assert env.actions_this_timestep[0][0] == pipe_name
        assert env.actions_this_timestep[0][1] == new_diameter
    else:
        assert len(env.actions_this_timestep) == 0
    
    # Test no-action (action 0)
    pipe_idx = env.current_pipe_index
    pipe_name = env.pipe_names[pipe_idx]
    initial_diameter = env.current_network.get_link(pipe_name).diameter
    
    obs, reward, terminated, truncated, info = env.step(0)
    assert len(env.actions_this_timestep) == 0 or env.actions_this_timestep[-1][0] != pipe_name

def test_network_metrics_with_upgrades(env):
    """Test how pressure deficit and demand satisfaction change with pipe upgrades."""
    # Initial network performance
    _, initial_metrics, _ = env._simulate_network(env.current_network)
    initial_pressure_deficit = initial_metrics.get('total_pressure_deficit', 0)
    initial_demand_satisfaction = initial_metrics.get('demand_satisfaction_ratio', 0)
    
    # Upgrade several pipes and measure impact
    # We'll upgrade first 5 pipes to largest diameter
    largest_diameter_action = len(env.pipe_diameter_options)  # Corresponds to the largest diameter
    
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(largest_diameter_action)
    
    # Simulate again to get updated metrics
    _, updated_metrics, _ = env._simulate_network(env.current_network)
    updated_pressure_deficit = updated_metrics.get('total_pressure_deficit', 0)
    updated_demand_satisfaction = updated_metrics.get('demand_satisfaction_ratio', 0)
    
    # Verify that upgrades have positive impact
    assert updated_pressure_deficit <= initial_pressure_deficit, "Pressure deficit should decrease or remain the same"
    assert updated_demand_satisfaction >= initial_demand_satisfaction, "Demand satisfaction should increase or remain the same"

def test_budget_management(env):
    """Test that budget is properly managed during pipe upgrades."""
    initial_budget = env.cumulative_budget
    
    # Upgrade a pipe with the largest diameter
    largest_diameter_action = len(env.pipe_diameter_options)
    
    # Take enough steps to complete one time step
    while env.current_pipe_index < len(env.pipe_names) - 1:
        obs, reward, terminated, truncated, info = env.step(largest_diameter_action)
    
    # Take one more step to finalize the time step and get budget update
    obs, reward, terminated, truncated, info = env.step(largest_diameter_action)
    
    # Budget should have decreased due to pipe upgrades
    assert env.cumulative_budget < initial_budget, "Budget should decrease after pipe upgrades"
    assert 'cost_of_intervention' in info, "Info dict should contain intervention cost"

def test_all_pipes_upgrade_visualization(env):
    """
    Test that systematically upgrades all pipes to the largest diameter
    and plots the changes in pressure deficit and demand satisfaction.
    """
    # Data collection structures
    pipe_upgrades = []
    pressure_deficits = []
    demand_satisfactions = []
    costs = []
    rewards = []
    
    # Initial metrics
    _, initial_metrics, _ = env._simulate_network(env.current_network)
    pressure_deficits.append(initial_metrics.get('total_pressure_deficit', 0))
    demand_satisfactions.append(initial_metrics.get('demand_satisfaction_ratio', 0))
    pipe_upgrades.append(0)
    costs.append(0)
    rewards.append(0)
    
    # Get the largest diameter action
    largest_diameter_action = len(env.pipe_diameter_options)
    
    # Upgrade all pipes systematically
    total_pipes = len(env.pipe_names)
    upgrade_count = 0
    
    # While we haven't terminated (i.e., reached end of episode)
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and upgrade_count < total_pipes:
        # Upgrade current pipe to largest diameter
        obs, reward, terminated, truncated, info = env.step(largest_diameter_action)
        
        # If we've processed all pipes for this time step, record metrics
        if 'pressure_deficit' in info:
            upgrade_count += len(env.actions_this_timestep)
            pipe_upgrades.append(upgrade_count)
            pressure_deficits.append(info.get('pressure_deficit', 0))
            demand_satisfactions.append(info.get('demand_satisfaction', 0))
            costs.append(info.get('cost_of_intervention', 0))
            rewards.append(info.get('step_reward', 0))
    
    # Create the figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot pressure deficit
    ax1.plot(pipe_upgrades, pressure_deficits, 'r-o', label='Pressure Deficit')
    ax1.set_xlabel('Number of Pipes Upgraded')
    ax1.set_ylabel('Pressure Deficit')
    ax1.set_title('Pressure Deficit vs Number of Pipe Upgrades')
    ax1.grid(True)
    ax1.legend()
    
    # Plot demand satisfaction
    ax2.plot(pipe_upgrades, demand_satisfactions, 'b-o', label='Demand Satisfaction Ratio')
    ax2.set_xlabel('Number of Pipes Upgraded')
    ax2.set_ylabel('Demand Satisfaction Ratio')
    ax2.set_title('Demand Satisfaction vs Number of Pipe Upgrades')
    ax2.grid(True)
    ax2.legend()
    
    # Plot costs and rewards
    ax3.plot(pipe_upgrades, costs, 'm-o', label='Intervention Cost')
    ax3.plot(pipe_upgrades, rewards, 'g-o', label='Step Reward')
    ax3.set_xlabel('Number of Pipes Upgraded')
    ax3.set_ylabel('Cost / Reward')
    ax3.set_title('Costs and Rewards vs Number of Pipe Upgrades')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "network_performance_with_upgrades.png"))
    
    # Also save the data to CSV for further analysis
    df = pd.DataFrame({
        'pipes_upgraded': pipe_upgrades,
        'pressure_deficit': pressure_deficits,
        'demand_satisfaction': demand_satisfactions,
        'intervention_cost': costs,
        'step_reward': rewards
    })
    df.to_csv(os.path.join(plots_dir, "network_performance_data.csv"), index=False)
    
    # For assertions, verify data is collected properly
    assert len(pipe_upgrades) > 1, "Should have collected data points"
    assert pressure_deficits[-1] <= pressure_deficits[0], "Pressure deficit should decrease with upgrades"

def test_selective_pipe_upgrades(env):
    """
    Test that selectively upgrades only the critical pipes
    and compares performance with full upgrade strategy.
    """
    # Reset to ensure consistent state
    env.reset(options={'scenario_name': HANOI_SCENARIOS[0]})
    
    # Data for storing performance
    selective_upgrades = {}
    
    # Get initial network pipe info
    pipe_lengths = {p: env.current_network.get_link(p).length for p in env.pipe_names}
    
    # Sort pipes by length (assuming longer pipes are more critical for testing purposes)
    # critical_pipes = sorted(pipe_lengths.items(), key=lambda x: x[1], reverse=True)[:10]
    # critical_pipe_names = [p[0] for p in critical_pipes]

    # Critical pipes are those connected to the reservoir
    critical_pipe_names = ['12', '11', '10', '2', '1', '21', '22']
    
    # Run through the environment, upgrading only critical pipes
    largest_diameter_action = len(env.pipe_diameter_options)
    
    while True:
        pipe_idx = env.current_pipe_index
        pipe_name = env.pipe_names[pipe_idx]
        
        # Upgrade only critical pipes
        if pipe_name in critical_pipe_names:
            action = largest_diameter_action
        else:
            action = 0  # No action for non-critical pipes
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record metrics at the end of a time step
        if 'pressure_deficit' in info:
            selective_upgrades[env.current_time_step - 1] = {
                'pressure_deficit': info.get('pressure_deficit', 0),
                'demand_satisfaction': info.get('demand_satisfaction', 0),
                'cost': info.get('cost_of_intervention', 0),
                'pipes_upgraded': len(env.actions_this_timestep)
            }
        
        if terminated or truncated:
            break
    
    # Save selective upgrade strategy results
    plots_dir = "Plots/test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.DataFrame.from_dict(selective_upgrades, orient='index')
    df.index.name = 'time_step'
    df.to_csv(os.path.join(plots_dir, "selective_upgrade_strategy.csv"))
    
    # Simple assertion to ensure we collected data
    assert len(selective_upgrades) > 0, "Should have collected performance data"

def test_all_pipes_upgrade_visualization_anytown(anytown_env):
    """
    Test that systematically upgrades all pipes in Anytown network to the largest diameter
    and plots the changes in pressure deficit and demand satisfaction.
    """
    # Data collection structures
    pipe_upgrades = []
    pressure_deficits = []
    demand_satisfactions = []
    costs = []
    rewards = []
    
    # Initial metrics
    _, initial_metrics, _ = anytown_env._simulate_network(anytown_env.current_network)
    pressure_deficits.append(initial_metrics.get('total_pressure_deficit', 0))
    demand_satisfactions.append(initial_metrics.get('demand_satisfaction_ratio', 0))
    pipe_upgrades.append(0)
    costs.append(0)
    rewards.append(0)
    
    # Get the largest diameter action
    largest_diameter_action = len(anytown_env.pipe_diameter_options)
    
    # Upgrade all pipes systematically
    total_pipes = len(anytown_env.pipe_names)
    upgrade_count = 0
    
    # While we haven't terminated (i.e., reached end of episode)
    terminated = False
    truncated = False

    while not (terminated or truncated) and upgrade_count < total_pipes:
        # Upgrade current pipe to largest diameter
        obs, reward, terminated, truncated, info = anytown_env.step(largest_diameter_action)
        
        # If we've processed all pipes for this time step, record metrics
        if 'pressure_deficit' in info:
            upgrade_count += len(anytown_env.actions_this_timestep)
            pipe_upgrades.append(upgrade_count)
            pressure_deficits.append(info.get('pressure_deficit', 0))
            demand_satisfactions.append(info.get('demand_satisfaction', 0))
            costs.append(info.get('cost_of_intervention', 0))
            rewards.append(info.get('step_reward', 0))
    
    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot pressure deficit
    ax1.plot(pipe_upgrades, pressure_deficits, 'r-o', label='Pressure Deficit')
    ax1.set_xlabel('Number of Pipes Upgraded')
    ax1.set_ylabel('Pressure Deficit')
    ax1.set_title('Pressure Deficit vs Number of Pipe Upgrades (Anytown)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot demand satisfaction
    ax2.plot(pipe_upgrades, demand_satisfactions, 'b-o', label='Demand Satisfaction Ratio')
    ax2.set_xlabel('Number of Pipes Upgraded')
    ax2.set_ylabel('Demand Satisfaction Ratio')
    ax2.set_title('Demand Satisfaction vs Number of Pipe Upgrades (Anytown)')
    ax2.grid(True)
    ax2.legend()

    # Plot costs and rewards
    ax3.plot(pipe_upgrades, costs, 'm-o', label='Intervention Cost')
    ax3.plot(pipe_upgrades, rewards, 'g-o', label='Step Reward')
    ax3.set_xlabel('Number of Pipes Upgraded')
    ax3.set_ylabel('Cost / Reward')
    ax3.set_title('Costs and Rewards vs Number of Pipe Upgrades (Anytown)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "anytown_network_performance_with_upgrades.png"))
    
    # Also save the data to CSV for further analysis
    df = pd.DataFrame({
        'pipes_upgraded': pipe_upgrades,
        'pressure_deficit': pressure_deficits,
        'demand_satisfaction': demand_satisfactions,
        'intervention_cost': costs,
        'step_reward': rewards
    })
    df.to_csv(os.path.join(plots_dir, "anytown_network_performance_data.csv"), index=False)
    
    # For assertions, verify data is collected properly
    assert len(pipe_upgrades) > 1, "Should have collected data points"
    assert pressure_deficits[-1] <= pressure_deficits[0], "Pressure deficit should decrease with upgrades"

def test_selective_pipe_upgrades_anytown(anytown_env):
    """
    Test that selectively upgrades only the critical pipes in Anytown network
    and compares performance with full upgrade strategy.
    """
    # Reset to ensure consistent state
    anytown_env.reset(options={'scenario_name': ANYTOWN_SCENARIOS[0]})
    
    # Data for storing performance
    selective_upgrades = {}
    
    # Get initial network pipe info
    pipe_lengths = {p: anytown_env.current_network.get_link(p).length for p in anytown_env.pipe_names}
    
    # For Anytown, let's define critical pipes as those connected to reservoirs/tanks or with large diameters
    # This is network specific and may need adjustment based on Anytown topology
    network = anytown_env.current_network
    
    # Find pipes connected to reservoirs or tanks
    critical_pipe_names = []
    for node_name, node in network.nodes():
        if node.node_type in ('Reservoir', 'Tank'):
            # Get pipes connected to this node
            for link_name, link in network.links():
                if link.start_node_name == node_name or link.end_node_name == node_name:
                    critical_pipe_names.append(link_name)

    # If too few critical pipes were found, add some based on diameter
    if len(critical_pipe_names) < 5:  
        # Find largest diameter pipes
        pipe_diameters = {p: network.get_link(p).diameter for p in anytown_env.pipe_names}
        sorted_pipes = sorted(pipe_diameters.items(), key=lambda x: x[1], reverse=True)
        additional_pipes = [p[0] for p in sorted_pipes[:10] if p[0] not in critical_pipe_names]
        critical_pipe_names.extend(additional_pipes)
    
    print(f"Selected {len(critical_pipe_names)} critical pipes for Anytown: {critical_pipe_names}")
    
    # Run through the environment, upgrading only critical pipes
    largest_diameter_action = len(anytown_env.pipe_diameter_options)
    
    while True:
        pipe_idx = anytown_env.current_pipe_index
        pipe_name = anytown_env.pipe_names[pipe_idx]
        
        # Upgrade only critical pipes
        if pipe_name in critical_pipe_names:
            action = largest_diameter_action
        else:
            action = 0  # No action for non-critical pipes
            
        obs, reward, terminated, truncated, info = anytown_env.step(action)
        
        # Record metrics at the end of a time step
        if 'pressure_deficit' in info:
            selective_upgrades[anytown_env.current_time_step - 1] = {
                'pressure_deficit': info.get('pressure_deficit', 0),
                'demand_satisfaction': info.get('demand_satisfaction', 0),
                'cost': info.get('cost_of_intervention', 0),
                'pipes_upgraded': len(anytown_env.actions_this_timestep)
            }

        if terminated or truncated:
            break
    
    # Save selective upgrade strategy results
    plots_dir = "Plots/test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.DataFrame.from_dict(selective_upgrades, orient='index')
    df.index.name = 'time_step'
    df.to_csv(os.path.join(plots_dir, "anytown_selective_upgrade_strategy.csv"))
    
    # Simple assertion to ensure we collected data
    assert len(selective_upgrades) > 0, "Should have collected performance data"

def test_compare_hanoi_vs_anytown():
    """
    Test that compares upgrade effects between Hanoi and Anytown networks.
    Loads saved results and creates comparison plots.
    """
    plots_dir = "Plots/test_plots"
    anytown_file = os.path.join(plots_dir, "anytown_network_performance_data.csv")
    hanoi_file = os.path.join(plots_dir, "network_performance_data.csv")
    
    # Check if both files exist, if not just pass the test
    if not (os.path.exists(anytown_file) and os.path.exists(hanoi_file)):
        pytest.skip("Data files for comparison not found. Run the individual tests first.")
    
    # Load the data
    anytown_df = pd.read_csv(anytown_file)
    hanoi_df = pd.read_csv(hanoi_file)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Normalize pipe upgrades as percentage of total pipes
    hanoi_df['pipes_upgraded_pct'] = hanoi_df['pipes_upgraded'] / hanoi_df['pipes_upgraded'].max() * 100
    anytown_df['pipes_upgraded_pct'] = anytown_df['pipes_upgraded'] / anytown_df['pipes_upgraded'].max() * 100
    
    # Comparison of pressure deficit reduction
    # Normalize pressure deficit (percentage of initial)
    hanoi_df['pressure_deficit_normalized'] = hanoi_df['pressure_deficit'] / hanoi_df['pressure_deficit'].iloc[0] * 100
    anytown_df['pressure_deficit_normalized'] = anytown_df['pressure_deficit'] / anytown_df['pressure_deficit'].iloc[0] * 100

    ax1.plot(hanoi_df['pipes_upgraded_pct'], hanoi_df['pressure_deficit_normalized'], 'r-o', label='Hanoi')
    ax1.plot(anytown_df['pipes_upgraded_pct'], anytown_df['pressure_deficit_normalized'], 'b-s', label='Anytown')
    ax1.set_xlabel('Percentage of Pipes Upgraded')
    ax1.set_ylabel('Pressure Deficit (% of Initial)')
    ax1.set_title('Pressure Deficit Reduction: Hanoi vs. Anytown')
    ax1.grid(True)
    ax1.legend()
    
    # Comparison of cost efficiency
    # Calculate cost per unit of pressure deficit reduction
    hanoi_df['pressure_deficit_reduction'] = hanoi_df['pressure_deficit'].iloc[0] - hanoi_df['pressure_deficit']
    anytown_df['pressure_deficit_reduction'] = anytown_df['pressure_deficit'].iloc[0] - anytown_df['pressure_deficit']
    
    hanoi_df['cost_per_pd_reduction'] = hanoi_df['intervention_cost'] / hanoi_df['pressure_deficit_reduction'].replace(0, np.nan)
    anytown_df['cost_per_pd_reduction'] = anytown_df['intervention_cost'] / anytown_df['pressure_deficit_reduction'].replace(0, np.nan)
    
    ax2.plot(hanoi_df['pipes_upgraded_pct'], hanoi_df['cost_per_pd_reduction'], 'r-o', label='Hanoi')
    ax2.plot(anytown_df['pipes_upgraded_pct'], anytown_df['cost_per_pd_reduction'], 'b-s', label='Anytown')
    ax2.set_xlabel('Percentage of Pipes Upgraded')
    ax2.set_ylabel('Cost per Unit Pressure Deficit Reduction')
    ax2.set_title('Cost-Efficiency: Hanoi vs. Anytown')
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale('log')  # Log scale makes comparison easier
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hanoi_vs_anytown_comparison.png"))
 
if __name__ == "__main__":
    # Run tests and generate plots
    pytest.main(["-xvs", __file__])