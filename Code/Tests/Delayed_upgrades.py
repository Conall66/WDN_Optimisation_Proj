
import os
import sys
import tempfile
import shutil
import wntr
import matplotlib.pyplot as plt
import pandas as pd

script = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Hydraulic_2 import run_epanet_simulation, evaluate_network_performance
from Reward2 import calculate_reward, compute_total_cost

def test_upgrade_strategies(network_name='hanoi'):
    """
    Compares two pipe upgrade strategies:
    1. Upgrading all pipes at once to a larger diameter
    2. Upgrading one pipe at a time (with increasing budget)
    
    Args:
        network_name: 'hanoi' or 'anytown'
    """
    print(f"\n=== COMPARING UPGRADE STRATEGIES FOR {network_name.upper()} NETWORK ===")
    
    
    # Define configurations
    PIPES_CONFIG = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    
    # Budget configurations
    BUDGET_CONFIG_HANOI = {
        "initial_budget_per_step": 1_000_000.0,
        "start_of_episode_budget": 5_000_000.0,
        "ongoing_debt_penalty_factor": 0.0001,
        "max_debt": 10_000_000.0,
        "labour_cost_per_meter": 100.0
    }
    
    BUDGET_CONFIG_ANYTOWN = {
        "initial_budget_per_step": 2_000_000.0,
        "start_of_episode_budget": 5_000_000.0,
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
    wn_base = wntr.network.WaterNetworkModel(base_inp)
    
    # Get all pipe IDs from the network
    pipe_ids = list(wn_base.pipe_name_list)
    num_pipes = len(pipe_ids)
    print(f"Network has {num_pipes} pipes")
    
    # Get unique diameters from PIPES_CONFIG (sorted from smallest to largest)
    unique_diameters = sorted([config['diameter'] for config in PIPES_CONFIG.values()])
    target_diameter = unique_diameters[-1]  # Use the largest diameter for upgrades
    print(f"Target upgrade diameter: {target_diameter:.4f}m")
    
    # Save initial diameters for later restoration
    initial_diameters = {pipe_id: wn_base.get_link(pipe_id).diameter for pipe_id in pipe_ids}
    
    # Get baseline network performance
    sim_results = run_epanet_simulation(wn_base)
    metrics = evaluate_network_performance(wn_base, sim_results)
    baseline_pd = metrics['total_pressure_deficit']
    baseline_ds = metrics['demand_satisfaction_ratio']
    
    print(f"Baseline pressure deficit: {baseline_pd:.2f}")
    print(f"Baseline demand satisfaction: {baseline_ds:.2f}")
    
    # Prepare results storage
    results = {
        'strategy': [],
        'step': [],
        'reward': [],
        'cost_component': [],
        'pd_component': [],
        'demand_component': [],
        'total_cost': [],
        'cumulative_cost': [],
        'pressure_deficit': [],
        'demand_satisfaction': [],
        'budget_used': [],
        'budget_available': [],
        'pipes_upgraded': []
    }
    
    # -------------------------------------------------------------------------
    # Strategy 1: Upgrade all pipes at once
    # -------------------------------------------------------------------------
    print("\nTesting Strategy 1: Upgrade all pipes at once...")
    
    # Create a copy of the base network
    wn_all_at_once = wntr.network.WaterNetworkModel(base_inp)
    
    # Define the actions: upgrade all pipes to the target diameter
    actions = [(pipe_id, target_diameter) for pipe_id in pipe_ids]
    
    # Apply actions
    for pipe_id, new_diameter in actions:
        pipe = wn_all_at_once.get_link(pipe_id)
        pipe.diameter = new_diameter
    
    # Run simulation
    sim_results = run_epanet_simulation(wn_all_at_once)
    metrics = evaluate_network_performance(wn_all_at_once, sim_results)
    
    # Calculate cost
    total_cost = compute_total_cost(
        actions=actions,
        pipes_config=PIPES_CONFIG,
        wn=wn_all_at_once,
        energy_cost=0.0,
        labour_cost_per_meter=budget_config['labour_cost_per_meter']
    )
    
    # Calculate reward
    params = {
        'metrics': metrics,
        'cost_of_intervention': total_cost,
        'max_cost_normalization': budget_config['max_debt'],
        'baseline_pressure_deficit': baseline_pd,
        'baseline_demand_satisfaction': baseline_ds
    }
    reward, components = calculate_reward('custom_normalized', params)
    
    # Record results
    results['strategy'].append('All at once')
    results['step'].append(1)
    results['reward'].append(reward)
    results['cost_component'].append(components['weighted_cost'])
    results['pd_component'].append(components['weighted_pd'])
    results['demand_component'].append(components['weighted_demand'])
    results['total_cost'].append(total_cost)
    results['cumulative_cost'].append(total_cost)
    results['pressure_deficit'].append(metrics['total_pressure_deficit'])
    results['demand_satisfaction'].append(metrics['demand_satisfaction_ratio'])
    results['budget_used'].append(total_cost)
    results['budget_available'].append(budget_config['start_of_episode_budget'])
    results['pipes_upgraded'].append(len(actions))
    
    print(f"All-at-once strategy results:")
    print(f"  Pipes upgraded: {len(actions)}")
    print(f"  Total cost: ${total_cost:,.2f}")
    print(f"  Pressure deficit: {metrics['total_pressure_deficit']:.2f}")
    print(f"  Demand satisfaction: {metrics['demand_satisfaction_ratio']:.4f}")
    print(f"  Reward: {reward:.4f}")
    
    # -------------------------------------------------------------------------
    # Strategy 2: Upgrade one pipe at a time with increasing budget
    # -------------------------------------------------------------------------
    print("\nTesting Strategy 2: Upgrade one pipe at a time...")
    
    # Start with a fresh network
    wn_incremental = wntr.network.WaterNetworkModel(base_inp)
    
    # Sort pipes by length for a deterministic upgrade order (you could use other criteria)
    pipe_lengths = [(pipe_id, wn_incremental.get_link(pipe_id).length) for pipe_id in pipe_ids]
    # sorted_pipes = sorted(pipe_lengths, key=lambda x: x[1], reverse=True)  # Start with longest pipes

    # Sort pipes by ID
    sorted_pipes = sorted(pipe_lengths, key=lambda x: x[0])  # Sort by pipe ID for consistency
    
    # Track budget
    available_budget = budget_config['start_of_episode_budget']
    cumulative_cost = 0
    pipes_upgraded_so_far = set()
    
    # Upgrade pipes one by one until budget is exhausted
    for step, (pipe_id, length) in enumerate(sorted_pipes, 1):
        # Calculate cost to upgrade this pipe
        pipe_action = [(pipe_id, target_diameter)]
        pipe_cost = compute_total_cost(
            actions=pipe_action,
            pipes_config=PIPES_CONFIG,
            wn=wn_incremental,
            energy_cost=0.0,
            labour_cost_per_meter=budget_config['labour_cost_per_meter']
        )
        
        # Check if we have budget for this pipe
        if pipe_cost > available_budget:
            print(f"  Step {step}: Cannot afford to upgrade pipe {pipe_id} (cost: ${pipe_cost:,.2f}, budget: ${available_budget:,.2f})")
            
            # Still record the state without making changes
            sim_results = run_epanet_simulation(wn_incremental)
            metrics = evaluate_network_performance(wn_incremental, sim_results)
            
            params = {
                'metrics': metrics,
                'cost_of_intervention': 0.0,  # No new cost for this step
                'max_cost_normalization': budget_config['max_debt'],
                'baseline_pressure_deficit': baseline_pd,
                'baseline_demand_satisfaction': baseline_ds
            }
            reward, components = calculate_reward('custom_normalized', params)
            
            results['strategy'].append('One at a time')
            results['step'].append(step)
            results['reward'].append(reward)
            results['cost_component'].append(components['weighted_cost'])
            results['pd_component'].append(components['weighted_pd'])
            results['demand_component'].append(components['weighted_demand'])
            results['total_cost'].append(0)  # No new cost
            results['cumulative_cost'].append(cumulative_cost)
            results['pressure_deficit'].append(metrics['total_pressure_deficit'])
            results['demand_satisfaction'].append(metrics['demand_satisfaction_ratio'])
            results['budget_used'].append(0)  # No budget used this step
            results['budget_available'].append(available_budget)
            results['pipes_upgraded'].append(len(pipes_upgraded_so_far))
            
            # Increase budget for next step
            available_budget += budget_config['initial_budget_per_step']
            continue
        
        # Upgrade the pipe
        wn_incremental.get_link(pipe_id).diameter = target_diameter
        pipes_upgraded_so_far.add(pipe_id)
        
        # Run simulation
        sim_results = run_epanet_simulation(wn_incremental)
        metrics = evaluate_network_performance(wn_incremental, sim_results)
        
        # Update costs and budget
        available_budget -= pipe_cost
        cumulative_cost += pipe_cost
        
        # Calculate reward
        params = {
            'metrics': metrics,
            'cost_of_intervention': pipe_cost,
            'max_cost_normalization': budget_config['max_debt'],
            'baseline_pressure_deficit': baseline_pd,
            'baseline_demand_satisfaction': baseline_ds
        }
        reward, components = calculate_reward('custom_normalized', params)
        
        # Record results
        results['strategy'].append('One at a time')
        results['step'].append(step)
        results['reward'].append(reward)
        results['cost_component'].append(components['weighted_cost'])
        results['pd_component'].append(components['weighted_pd'])
        results['demand_component'].append(components['weighted_demand'])
        results['total_cost'].append(pipe_cost)
        results['cumulative_cost'].append(cumulative_cost)
        results['pressure_deficit'].append(metrics['total_pressure_deficit'])
        results['demand_satisfaction'].append(metrics['demand_satisfaction_ratio'])
        results['budget_used'].append(pipe_cost)
        results['budget_available'].append(available_budget)
        results['pipes_upgraded'].append(len(pipes_upgraded_so_far))
        
        print(f"  Step {step}: Upgraded pipe {pipe_id} (cost: ${pipe_cost:,.2f}, budget left: ${available_budget:,.2f})")
        print(f"    Pressure deficit: {metrics['total_pressure_deficit']:.2f}, Demand satisfaction: {metrics['demand_satisfaction_ratio']:.4f}")
        print(f"    Reward: {reward:.4f}")
        
        # Increase budget for next step
        available_budget += budget_config['initial_budget_per_step']
        
        # Stop if we've upgraded all pipes
        if len(pipes_upgraded_so_far) == len(pipe_ids):
            print(f"  All pipes upgraded after {step} steps.")
            break
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(14, 20))
    
    # Plot 1: Reward Comparison
    plt.subplot(4, 1, 1)
    one_at_a_time = df[df['strategy'] == 'One at a time']
    all_at_once = df[df['strategy'] == 'All at once']
    
    plt.plot(one_at_a_time['step'], one_at_a_time['reward'], 'o-', label='One at a time', linewidth=2)
    plt.axhline(y=all_at_once['reward'].values[0], color='r', linestyle='--', 
                label=f'All at once ({all_at_once["reward"].values[0]:.4f})')
    
    plt.title(f'Reward Comparison - {network_name.capitalize()} Network', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Reward Components for One-at-a-time
    plt.subplot(4, 1, 2)
    plt.plot(one_at_a_time['step'], one_at_a_time['reward'], 'o-', label='Total Reward', linewidth=2)
    plt.plot(one_at_a_time['step'], one_at_a_time['cost_component'], 's--', label='Cost Component')
    plt.plot(one_at_a_time['step'], one_at_a_time['pd_component'], '^--', label='Pressure Deficit Component')
    plt.plot(one_at_a_time['step'], one_at_a_time['demand_component'], 'd--', label='Demand Satisfaction Component')
    
    plt.title(f'Reward Components (One-at-a-time) - {network_name.capitalize()}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Component Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Hydraulic Performance Metrics
    plt.subplot(4, 1, 3)
    plt.plot(one_at_a_time['step'], one_at_a_time['pressure_deficit'], 'o-', label='Pressure Deficit', color='red')
    plt.ylabel('Pressure Deficit', color='red', fontsize=12)
    plt.tick_params(axis='y', labelcolor='red')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.gca().twinx()
    ax2.plot(one_at_a_time['step'], one_at_a_time['demand_satisfaction'], 's-', label='Demand Satisfaction', color='green')
    ax2.set_ylabel('Demand Satisfaction', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add reference lines for all-at-once strategy
    plt.axhline(y=all_at_once['pressure_deficit'].values[0], color='r', linestyle=':', alpha=0.7,
               label=f'All-at-once PD ({all_at_once["pressure_deficit"].values[0]:.2f})')
    ax2.axhline(y=all_at_once['demand_satisfaction'].values[0], color='g', linestyle=':', alpha=0.7,
               label=f'All-at-once DS ({all_at_once["demand_satisfaction"].values[0]:.4f})')
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f'Hydraulic Metrics - {network_name.capitalize()}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    
    # Plot 4: Budget and Cost Comparison
    plt.subplot(4, 1, 4)
    plt.bar(one_at_a_time['step'], one_at_a_time['budget_used'], color='blue', alpha=0.5, label='Step Cost')
    plt.plot(one_at_a_time['step'], one_at_a_time['cumulative_cost'], 'o-', color='navy', label='Cumulative Cost')
    plt.axhline(y=all_at_once['total_cost'].values[0], color='r', linestyle='--', 
               label=f'All-at-once Cost (${all_at_once["total_cost"].values[0]:,.0f})')
    
    plt.title(f'Budget and Cost - {network_name.capitalize()}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'Plots/{network_name}_upgrade_strategies_comparison.png', dpi=150)
    plt.show()
    
    print(f"\nResults saved to {network_name}_upgrade_strategies_comparison.png")
    
    # Final comparison
    final_one_at_time = one_at_a_time.iloc[-1]
    all_at_once_result = all_at_once.iloc[0]
    
    print("\n=== FINAL COMPARISON ===")
    print(f"Strategy 1 (All at once):")
    print(f"  Total cost: ${all_at_once_result['total_cost']:,.2f}")
    print(f"  Final reward: {all_at_once_result['reward']:.4f}")
    print(f"  Pressure deficit: {all_at_once_result['pressure_deficit']:.2f}")
    print(f"  Demand satisfaction: {all_at_once_result['demand_satisfaction']:.4f}")
    
    print(f"\nStrategy 2 (One at a time):")
    print(f"  Total cost: ${final_one_at_time['cumulative_cost']:,.2f}")
    print(f"  Final reward: {final_one_at_time['reward']:.4f}")
    print(f"  Pressure deficit: {final_one_at_time['pressure_deficit']:.2f}")
    print(f"  Demand satisfaction: {final_one_at_time['demand_satisfaction']:.4f}")
    print(f"  Pipes upgraded: {final_one_at_time['pipes_upgraded']} out of {len(pipe_ids)}")
    
    return df

if __name__ == "__main__":

    # Create a temporary directory for plots
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'Plots'), exist_ok=True)
    
    try:
        # Run tests for both networks
        hanoi_results = test_upgrade_strategies('hanoi')
        anytown_results = test_upgrade_strategies('anytown')
        
        # Save results to CSV files
        hanoi_results.to_csv(os.path.join(temp_dir, 'hanoi_upgrade_strategies.csv'), index=False)
        anytown_results.to_csv(os.path.join(temp_dir, 'anytown_upgrade_strategies.csv'), index=False)
        
        print(f"\nResults saved to {temp_dir}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)