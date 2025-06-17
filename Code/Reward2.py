"""
This script defines the reward calculation logic for the DRL agent.

This version is REFACTORED to:
1.  Implement a custom, state-relative normalization scheme for reward components.
2.  Ensure the final reward is always clamped between 0 and 1.
3.  Consolidate reward logic into a single clear function.
"""
from typing import Dict, List, Any
import wntr

# ===================================================================
# 1. CORE REWARD STRATEGY (NEW)
# ===================================================================

def _reward_custom_normalized(params: Dict[str, Any]) -> tuple[float, dict]:
    """
    REWARD STRATEGY: A custom weighted reward function with dynamic normalization.

    - Pressure Deficit (PD): Normalized relative to the episode's starting PD.
    - Demand Satisfaction (DS): Normalized relative to the episode's starting DS.
    - Cost: Normalized against a predefined maximum cost.
    - Budget: Apply penalty if current budget becomes negative after intervention

    The final reward is a weighted sum of these components, clipped to be [0, 1].
    """
    metrics = params['metrics']
    max_cost = params.get('max_cost_normalization', 1.0)
    baseline_pd = params.get('baseline_pressure_deficit', 0.0)
    baseline_ds = params.get('baseline_demand_satisfaction', 0.0)
    
    # Get budget information
    current_budget = params.get('current_budget', float('inf'))
    cost_of_intervention = params.get('cost_of_intervention', 0.0)
    budget_after_intervention = current_budget - cost_of_intervention

    # --- Component 1: Pressure Deficit (PD) Normalization ---
    # Goal: 1.0 for zero PD, 0.0 for baseline PD.
    current_pd = metrics.get('total_pressure_deficit', baseline_pd)
    pd_range = baseline_pd
    if pd_range > 0:
        # Lower PD is better. We invert it so 1.0 is best.
        pd_component = 1.0 - (min(current_pd, baseline_pd) / pd_range)
    else:
        # If baseline PD is already 0, any PD is bad.
        pd_component = 1.0 if current_pd <= 0 else 0.0
    
    pd_component = max(0, min(1, pd_component)) # Clamp to [0, 1]

    # --- Component 2: Demand Satisfaction (DS) Normalization ---
    # Goal: 1.0 for full satisfaction (1.0), 0.0 for baseline DS.
    current_ds = metrics.get('demand_satisfaction_ratio', baseline_ds)
    ds_range = 1.0 - baseline_ds
    if ds_range > 0:
        # Higher DS is better.
        ds_component = (current_ds - baseline_ds) / ds_range
    else:
        # If baseline DS is already 1.0, any drop is bad.
        ds_component = 1.0 if current_ds >= 1.0 else 0.0
        
    ds_component = max(0, min(1, ds_component)) # Clamp to [0, 1]

    # --- Component 3: Cost Normalization ---
    # Goal: 1.0 for zero cost, 0.0 for max_cost.
    if max_cost > 0:
        cost_component = 1.0 - (min(cost_of_intervention, max_cost) / max_cost)
    else:
        cost_component = 1.0 if cost_of_intervention <= 0 else 0.0

    cost_component = max(0, min(1, cost_component)) # Clamp to [0, 1]

    # --- Combine Components with Equal Weights ---
    w_cost, w_pd, w_demand = 0.4, 0.3, 0.3
    
    reward = (w_cost * cost_component) + (w_pd * pd_component) + (w_demand * ds_component)
    
    # Apply budget penalty directly in the reward function
    if budget_after_intervention < 0:
        budget_penalty_factor = 0.1  # Harsh penalty for exceeding budget
        reward = reward * budget_penalty_factor
    
    # Final clamp to ensure reward is strictly within [0, 1]
    final_reward = max(0, min(1, reward))

    components = {
        'weighted_cost': w_cost * cost_component,
        'weighted_pd': w_pd * pd_component,
        'weighted_demand': w_demand * ds_component,
        'budget_after_intervention': budget_after_intervention,
        'budget_penalty_applied': budget_after_intervention < 0
    }
    return final_reward, components

# ===================================================================
# 2. PUBLIC API FUNCTIONS (Call these from the environment)
# ===================================================================

def calculate_reward(mode: str, params: Dict[str, Any]) -> tuple[float, dict]:
    """
    Dispatcher function to calculate reward based on the specified mode.
    NOTE: This is now simplified to primarily use the 'custom_normalized' mode.
    
    Args:
        mode: The reward strategy to use.
        params: A dictionary containing all necessary data for reward calculation.

    Returns:
        A tuple containing the final reward value and a dictionary of its components.
    """
    if mode == 'custom_normalized':
        return _reward_custom_normalized(params)
    else:
        raise ValueError(f"Invalid reward mode: '{mode}'. This version is designed for 'custom_normalized'.")

def compute_total_cost(
    actions: List[tuple],
    pipes_config: Dict,
    wn: wntr.network.WaterNetworkModel,
    energy_cost: float,
    labour_cost_per_meter: float = 100.0
) -> float:
    """
    Computes the total cost of interventions (pipe upgrades + energy + labour).
    (No changes to this function)
    """
    pipe_upgrade_cost = 0.0
    total_labour_cost = 0.0

    # Create a lookup for unit costs for faster access
    diameter_to_cost_map = {p['diameter']: p['unit_cost'] for p in pipes_config.values()}

    for pipe_id, new_diameter in actions:
        try:
            pipe_length = wn.get_link(pipe_id).length
            unit_cost = diameter_to_cost_map.get(new_diameter, 0)
            
            pipe_upgrade_cost += unit_cost * pipe_length
            total_labour_cost += labour_cost_per_meter * pipe_length
        except KeyError:
            print(f"Warning: Pipe ID '{pipe_id}' from actions not found in the current network model. Skipping cost calculation for it.")

    total_cost = pipe_upgrade_cost + total_labour_cost + energy_cost
    return total_cost

# ==================================
# 3. TEST FUNCTIONS (For unit testing)
# ==================================

import os
import wntr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the reward functions
from Reward2 import calculate_reward, compute_total_cost

# ...existing code...

def test_reward_profile(network_name='hanoi'):
    """
    Tests the reward profile for different pipe upgrade scenarios:
    1. No changes to the network (baseline)
    2. All pipes upgraded incrementally (uniform diameter profiles)
    
    Args:
        network_name: 'hanoi' or 'anytown'
    """
    print(f"\n=== TESTING REWARD PROFILE FOR {network_name.upper()} NETWORK ===")

    from Hydraulic_2 import run_epanet_simulation, evaluate_network_performance
    
    # Define configurations
    PIPES_CONFIG = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    
    # Budget configurations from Train_w_Plots2.py
    BUDGET_CONFIG_HANOI = {
        "initial_budget_per_step": 500_000.0,
        "start_of_episode_budget": 1_000_000.0,
        "ongoing_debt_penalty_factor": 0.0001,
        "max_debt": 10_000_000.0,
        "labour_cost_per_meter": 100.0
    }
    
    BUDGET_CONFIG_ANYTOWN = {
        "initial_budget_per_step": 2_000_000.0,
        "start_of_episode_budget": 4_000_000.0,
        "ongoing_debt_penalty_factor": 0.0001,
        "max_debt": 10_000_000.0,
        "labour_cost_per_meter": 100.0
    }
    
    # Select the appropriate configuration and base .inp file
    if network_name.lower() == 'hanoi':
        base_inp = os.path.join('Networks2', 'hanoi_densifying_1', 'Step_1.inp')
        budget_config = BUDGET_CONFIG_HANOI
    else:  # anytown
        base_inp = os.path.join('Networks2', 'anytown_densifying_1', 'Step_1.inp')
        budget_config = BUDGET_CONFIG_ANYTOWN
    
    # Load the network
    wn = wntr.network.WaterNetworkModel(base_inp)
    
    # Get all pipe IDs from the network
    pipe_ids = list(wn.pipe_name_list)
    num_pipes = len(pipe_ids)
    print(f"Network has {num_pipes} pipes")
    
    # Save initial diameters for later restoration
    initial_diameters = {pipe_id: wn.get_link(pipe_id).diameter for pipe_id in pipe_ids}
    
    # Get unique diameters from PIPES_CONFIG (sorted from smallest to largest)
    unique_diameters = sorted([config['diameter'] for config in PIPES_CONFIG.values()])
    num_diameters = len(unique_diameters)
    print(f"Available pipe diameters: {unique_diameters}")
    
    # For storing results
    results = {
        'scenario': [],
        'reward': [],
        'cost_component': [],
        'pd_component': [],
        'demand_component': [],
        'total_cost': [],
        'pressure_deficit': [],
        'demand_satisfaction': [],
        'budget_consumed_percent': []  # New field to track budget consumption
    }
    
    # --- 1. Run baseline scenario (no changes) ---
    print("\nRunning baseline scenario (no changes)...")
    
    # Simulate the base network using your hydraulic function
    sim_results = run_epanet_simulation(wn)
    
    # Calculate metrics using your hydraulic function
    metrics = evaluate_network_performance(wn, sim_results)
    
    # Extract the values
    pressure_deficit = metrics['total_pressure_deficit']
    demand_satisfaction = metrics['demand_satisfaction_ratio']
    
    print(f"Pressure deficit: {pressure_deficit:.2f}")
    print(f"Demand satisfaction: {demand_satisfaction:.2f}")
    
    # Store baseline metrics for normalization in reward function
    baseline_pd = pressure_deficit
    baseline_ds = demand_satisfaction
    
    # Calculate reward for baseline (no interventions)
    params = {
        'metrics': {
            'total_pressure_deficit': pressure_deficit,
            'demand_satisfaction_ratio': demand_satisfaction
        },
        'cost_of_intervention': 0.0,  # No changes = zero cost
        'max_cost_normalization': budget_config['max_debt'],
        'baseline_pressure_deficit': baseline_pd,
        'baseline_demand_satisfaction': baseline_ds
    }
    
    reward, components = calculate_reward('custom_normalized', params)
    
    # Record results
    results['scenario'].append('Baseline (No Changes)')
    results['reward'].append(reward)
    results['cost_component'].append(components['weighted_cost'])
    results['pd_component'].append(components['weighted_pd'])
    results['demand_component'].append(components['weighted_demand'])
    results['total_cost'].append(0)
    results['pressure_deficit'].append(pressure_deficit)
    results['demand_satisfaction'].append(demand_satisfaction)
    results['budget_consumed_percent'].append(0)  # No budget consumed
    
    print(f"Baseline results:")
    print(f"  Pressure Deficit: {pressure_deficit:.2f}")
    print(f"  Demand Satisfaction: {demand_satisfaction:.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Components: {components}")
    
    # --- 2. Test incremental uniform upgrades ---
    for d_idx, diameter in enumerate(unique_diameters):
        scenario_name = f"All pipes {diameter:.4f}m"
        print(f"\nTesting {scenario_name}...")
        
        # Reset network to initial state
        wn = wntr.network.WaterNetworkModel(base_inp)
        
        # Create action list to upgrade all pipes to this diameter
        actions = [(pipe_id, diameter) for pipe_id in pipe_ids]
        
        # Apply actions (upgrade all pipes to current diameter)
        for pipe_id, new_diameter in actions:
            pipe = wn.get_link(pipe_id)
            pipe.diameter = new_diameter
        
        sim_results = run_epanet_simulation(wn)
        
        # Calculate metrics using your hydraulic function
        metrics = evaluate_network_performance(wn, sim_results)
        
        # Extract the values
        pressure_deficit = metrics['total_pressure_deficit']
        demand_satisfaction = metrics['demand_satisfaction_ratio']
        
        # Calculate cost of upgrades
        pipe_upgrade_cost = compute_total_cost(
            actions=actions,
            pipes_config=PIPES_CONFIG,
            wn=wn,
            energy_cost=0.26,  # Assume no energy cost for this test
            labour_cost_per_meter=budget_config['labour_cost_per_meter']
        )
        
        # Calculate budget consumption as percentage
        budget_consumed_percent = (pipe_upgrade_cost / budget_config['max_debt']) * 100
        
        # Calculate reward
        params = {
            'metrics': {
                'total_pressure_deficit': pressure_deficit,
                'demand_satisfaction_ratio': demand_satisfaction
            },
            'cost_of_intervention': pipe_upgrade_cost,
            'max_cost_normalization': budget_config['max_debt'],
            'baseline_pressure_deficit': baseline_pd,
            'baseline_demand_satisfaction': baseline_ds
        }
        
        reward, components = calculate_reward('custom_normalized', params)
        
        # Record results
        results['scenario'].append(scenario_name)
        results['reward'].append(reward)
        results['cost_component'].append(components['weighted_cost'])
        results['pd_component'].append(components['weighted_pd'])
        results['demand_component'].append(components['weighted_demand'])
        results['total_cost'].append(pipe_upgrade_cost)
        results['pressure_deficit'].append(pressure_deficit)
        results['demand_satisfaction'].append(demand_satisfaction)
        results['budget_consumed_percent'].append(budget_consumed_percent)  # Add budget consumption
        
        print(f"  Pressure Deficit: {pressure_deficit:.2f}")
        print(f"  Demand Satisfaction: {demand_satisfaction:.2f}")
        print(f"  Cost: {pipe_upgrade_cost:.2f} ({budget_consumed_percent:.2f}% of max budget)")
        print(f"  Reward: {reward:.4f}")
        print(f"  Components: {components}")
    
    # Create DataFrame and visualize results
    df = pd.DataFrame(results)
    
    # Plotting - Now with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot reward and components
    ax1.plot(df['scenario'], df['reward'], 'o-', label='Total Reward', linewidth=2)
    ax1.plot(df['scenario'], df['cost_component'], 's--', label='Cost Component')
    ax1.plot(df['scenario'], df['pd_component'], '^--', label='Pressure Deficit Component')
    ax1.plot(df['scenario'], df['demand_component'], 'd--', label='Demand Satisfaction Component')
    
    ax1.set_title(f'Reward Profile for {network_name.capitalize()} Network')
    ax1.set_ylabel('Reward Component Value')
    ax1.grid(True)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot hydraulic metrics
    ax2.plot(df['scenario'], df['pressure_deficit'], 'o-', label='Pressure Deficit', color='red')
    ax2.set_ylabel('Pressure Deficit', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['scenario'], df['demand_satisfaction'], 's-', label='Demand Satisfaction', color='green')
    ax2_twin.set_ylabel('Demand Satisfaction', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    # Add legend for the second plot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_title(f'Hydraulic Metrics for {network_name.capitalize()} Network')
    
    # NEW: Plot budget consumption
    ax3.bar(df['scenario'], df['budget_consumed_percent'], color='blue', alpha=0.7)
    ax3.set_title(f'Budget Consumption for {network_name.capitalize()} Network')
    ax3.set_ylabel('Budget Consumed (%)')
    ax3.grid(True, axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add threshold lines for different budget limits
    ax3.axhline(y=(budget_config['start_of_episode_budget']/budget_config['max_debt'])*100, 
                color='green', linestyle='--', 
                label=f'Episode Start Budget: {budget_config["start_of_episode_budget"]:,.0f}')
    
    ax3.axhline(y=100, color='red', linestyle='--', 
                label=f'Max Debt: {budget_config["max_debt"]:,.0f}')
    
    # Add cost values as text on top of bars
    for i, cost in enumerate(df['total_cost']):
        if cost > 0:  # Only add text for non-zero costs
            ax3.text(i, df['budget_consumed_percent'][i] + 1, f'{cost:,.0f}', 
                    ha='center', va='bottom', rotation=0, fontsize=8)
    
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'Plots/{network_name}_reward_profile_test.png', dpi=150)
    plt.show()
    
    print(f"\nResults saved to {network_name}_reward_profile_test.png")
    return df

if __name__ == "__main__":
    # Run the test for both networks
    test_reward_profile('hanoi')
    test_reward_profile('anytown')

# Test reward calculation
# if __name__ == "__main__":
#     # Example usage
#     params = {
#         'metrics': {
#             'total_pressure_deficit': 50.0,
#             'demand_satisfaction_ratio': 0.8,
#         },
#         'cost_of_intervention': 2000.0,
#         'max_pd_normalization': 100.0,
#         'max_cost_normalization': 5000.0,
#     }

#     reward, components = calculate_reward('custom_normalization', params)
#     print(f"Reward: {reward}, Components: {components}")