
"""

In this file, we parse the same training environments as the DRL agent to generate an optimal pipe sizing configuration. We plot the computational cost compared with the deployed DRL agent and their relative performance/reward functions.

"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import pygad

# Import functions and classes from your existing project files
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import calculate_reward, compute_total_cost
from PPO_Environment import WNTRGymEnv # For DRL evaluation
from Actor_Critic_Nets2 import GraphPPOAgent # For DRL evaluation

# --- Configuration (can be adjusted) ---
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}
PIPE_DIAMETER_OPTIONS = sorted([p['diameter'] for p in PIPES_CONFIG.values()]) # Ensure sorted
LABOUR_COST = 100  # Ensure this matches the value in WNTRGymEnv
NETWORKS_FOLDER_PATH = 'Modified_nets'

# --- Global variables for GA fitness context (simplifies PyGAD integration) ---
ga_current_scenario_base_inp_path = None
ga_current_scenario_original_diameters = None
ga_current_scenario_max_pd = None
ga_simulation_calls_counter = 0

def calculate_max_pd_for_scenario(base_inp_path, pipes_config_dict):
    """
    Calculates the maximum pressure deficit for a given base network
    by setting all pipes to their minimum diameter.
    Returns: Tuple (max_pd, number_of_simulation_calls_for_this_calc)
    """
    try:
        wn_temp = wntr.network.WaterNetworkModel(base_inp_path)
        min_diameter_val = min(p['diameter'] for p in pipes_config_dict.values())
        for p_name in wn_temp.pipe_name_list:
            wn_temp.get_link(p_name).diameter = min_diameter_val
        
        temp_results, _ = run_epanet_simulation(wn_temp) 
        if temp_results and not temp_results.node['pressure'].isnull().values.any():
            temp_metrics = evaluate_network_performance(wn_temp, temp_results)
            return temp_metrics.get('total_pressure_deficit', float('inf')), 1
        return float('inf'), 1 # Count sim call even if it fails
    except Exception as e:
        print(f"Error in calculate_max_pd_for_scenario for {base_inp_path}: {e}")
        return float('inf'), 1


def ga_fitness_function(ga_instance, solution, solution_idx):
    """
    Fitness function for the Genetic Algorithm.
    The 'solution' is a list of indices, where each index corresponds to a diameter
    in PIPE_DIAMETER_OPTIONS for a respective pipe.
    """
    global ga_simulation_calls_counter
    ga_simulation_calls_counter += 1

    # Create a trial network from the base .inp file for the current scenario
    wn_trial = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
    
    actions_for_this_solution = []
    for i, pipe_name in enumerate(wn_trial.pipe_name_list):
        # solution[i] is an index for PIPE_DIAMETER_OPTIONS
        chosen_diameter_index = int(solution[i])
        new_diameter = PIPE_DIAMETER_OPTIONS[chosen_diameter_index]
        
        # Check if it's different from original for cost calculation
        original_diameter = ga_current_scenario_original_diameters[pipe_name]
        if abs(new_diameter - original_diameter) > 1e-6: # float comparison
            actions_for_this_solution.append((pipe_name, new_diameter))
        wn_trial.get_link(pipe_name).diameter = new_diameter

    sim_results, sim_metrics = run_epanet_simulation(wn_trial)

    if not sim_results or (hasattr(sim_results.node, 'pressure') and sim_results.node['pressure'].isnull().values.any()):
        return -1e9 # Very low fitness for failed/unstable simulation

    # Determine if any pipes were "downgraded" relative to the original .inp state
    downgraded_in_ga_solution = False
    for pipe_name, new_diam in actions_for_this_solution:
        if new_diam < ga_current_scenario_original_diameters[pipe_name]:
            downgraded_in_ga_solution = True
            break
            
    reward_val, _, _, _, _, _, _ = calculate_reward(
        current_network=wn_trial,
        original_pipe_diameters=ga_current_scenario_original_diameters,
        actions=actions_for_this_solution,
        pipes=PIPES_CONFIG,
        performance_metrics=sim_metrics,
        labour_cost=LABOUR_COST,
        downgraded_pipes=downgraded_in_ga_solution, 
        disconnections=False, # Assuming GA does not model disconnections for this setup
        max_pd=ga_current_scenario_max_pd
    )
    return float(reward_val)


def evaluate_drl_agent_on_final_state(drl_model_path, target_scenario_name, pipes_config_dict, labour_cost_val):
    """
    Evaluates the DRL agent by running it through the entire target_scenario_name
    and then assessing its final chosen pipe configuration for the last timestep of that scenario.
    Returns a dictionary of metrics for the DRL's final configuration.
    """
    print(f"  Evaluating DRL on full evolution of {target_scenario_name} to get final state...")
    drl_sim_calls_for_this_eval = 0

    # --- Determine the final .inp file and its original properties ---
    scenario_path = os.path.join(NETWORKS_FOLDER_PATH, target_scenario_name)
    inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])
    if not inp_files:
        raise FileNotFoundError(f"No .inp files found for DRL scenario {target_scenario_name}")
    
    final_inp_file_for_scenario = inp_files[-1]
    final_inp_path = os.path.join(scenario_path, final_inp_file_for_scenario)

    wn_final_base = wntr.network.WaterNetworkModel(final_inp_path)
    original_diameters_of_final_inp = {
        p_name: wn_final_base.get_link(p_name).diameter for p_name in wn_final_base.pipe_name_list
    }
    
    max_pd_for_final_inp, sim_calls_max_pd = calculate_max_pd_for_scenario(final_inp_path, pipes_config_dict)
    drl_sim_calls_for_this_eval += sim_calls_max_pd

    # --- Run DRL agent through the entire scenario ---
    # The WNTRGymEnv is set up to run only the target_scenario_name
    eval_env = WNTRGymEnv(pipes=pipes_config_dict, scenarios=[target_scenario_name], networks_folder=NETWORKS_FOLDER_PATH)
    
    # Temporarily modify WNTRGymEnv's simulate_network to count calls for this evaluation
    # This is a bit intrusive but effective for accurate counting here.
    original_env_sim_net = eval_env.simulate_network 
    # This counter will be specific to this DRL evaluation run
    # It will be incremented by the wrapped simulate_network
    # Needs to be a list or dict to be mutable inside the wrapper
    # Or, rely on the fact that WNTRGymEnv.step calls simulate_network a known number of times.
    
    # Let's count based on WNTRGymEnv structure:
    # reset -> _load_network_for_timestep -> simulate_network (1 call)
    # each step (if all pipes decided) -> simulate_network (main) + simulate_network (max_pd)
    # + _load_network_for_timestep (if not last) -> simulate_network
    
    drl_agent = GraphPPOAgent(eval_env, pipes_config_dict) # Dummy env for agent creation
    drl_agent.load(drl_model_path)

    obs, _ = eval_env.reset(options={'scenario': target_scenario_name}) # Force scenario
    drl_sim_calls_for_this_eval += 1 # For the simulate_network in _load_network_for_timestep during reset

    done = False
    final_drl_network_configuration = None
    actions_taken_by_drl_for_final_step = []

    num_scenario_timesteps = len(inp_files)

    while not done:
        action, _ = drl_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = eval_env.step(action)
        done = terminated or truncated

        # Count simulation calls within the step
        # 1 for main sim, 1 for max_pd sim (if results from main sim were good)
        drl_sim_calls_for_this_eval += 1 # main simulation in step
        if 'reward' in step_info : # Implies main sim was successful and max_pd was attempted
             drl_sim_calls_for_this_eval +=1 # max_pd simulation in step
        
        if not done: # If not done, _load_network_for_timestep was called
            drl_sim_calls_for_this_eval += 1

        if eval_env.current_time_step == num_scenario_timesteps -1 and eval_env.current_pipe_index == 0:
             # This is the start of decisions for the final .inp file.
             # The actions_this_timestep from the *previous* step were for the second to last .inp.
             # We need the actions applied *during* the processing of the final .inp.
             # This is tricky because actions_this_timestep is cleared.
             pass # Placeholder for thought

        if done:
            # eval_env.current_network is the network state *after* DRL's actions on the final .inp file.
            final_drl_network_configuration = eval_env.current_network 
            # The actions that transformed the *original final .inp* to this state:
            for p_name in final_drl_network_configuration.pipe_name_list:
                drl_chosen_diameter = final_drl_network_configuration.get_link(p_name).diameter
                original_final_inp_diam = original_diameters_of_final_inp[p_name]
                if abs(drl_chosen_diameter - original_final_inp_diam) > 1e-6:
                    actions_taken_by_drl_for_final_step.append((p_name, drl_chosen_diameter))
            break 
            
    eval_env.close()

    if final_drl_network_configuration is None:
        print(f"  DRL agent did not complete scenario {target_scenario_name} to yield a final configuration.")
        return {'reward': -1e10, 'cost': float('inf'), 'pd': float('inf'), 'demand_sat': 0, 
                'sim_calls': drl_sim_calls_for_this_eval, 
                'network_size': (0,0)}

    # --- Evaluate the DRL's final chosen configuration ---
    print(f"  Evaluating DRL's final network configuration for {final_inp_file_for_scenario}...")
    # This simulation is to get metrics for the *already configured* network by DRL.
    # It's an evaluation call, not part of DRL's decision process for *this* specific comparison.
    # However, to make it comparable to GA's "best solution evaluation", we can count it.
    drl_sim_calls_for_this_eval += 1
    final_drl_results, final_drl_metrics = run_epanet_simulation(final_drl_network_configuration)
    
    if not final_drl_results or (hasattr(final_drl_results.node, 'pressure') and final_drl_results.node['pressure'].isnull().values.any()):
        print(f"  DRL's final configuration for {target_scenario_name} is unstable.")
        return {'reward': -1e9, 'cost': float('inf'), 'pd': float('inf'), 'demand_sat': 0, 
                'sim_calls': drl_sim_calls_for_this_eval, 
                'network_size': (len(final_drl_network_configuration.node_name_list), len(final_drl_network_configuration.pipe_name_list))}

    # DRL is constrained to upgrades, so downgraded_pipes is False
    drl_reward, drl_cost, _, drl_demand_sat, _, _, _ = calculate_reward(
        current_network=final_drl_network_configuration,
        original_pipe_diameters=original_diameters_of_final_inp, # Compare against the base final .inp
        actions=actions_taken_by_drl_for_final_step,
        pipes=pipes_config_dict,
        performance_metrics=final_drl_metrics,
        labour_cost=labour_cost_val,
        downgraded_pipes=False, 
        disconnections=False,
        max_pd=max_pd_for_final_inp 
    )
    
    drl_abs_pd = final_drl_metrics.get('total_pressure_deficit', float('inf'))
    
    return {
        'reward': drl_reward,
        'cost': drl_cost,
        'pd': drl_abs_pd,
        'demand_sat': drl_demand_sat, # This is already a ratio
        'sim_calls': drl_sim_calls_for_this_eval,
        'network_size': (len(final_drl_network_configuration.node_name_list), len(final_drl_network_configuration.pipe_name_list))
    }


def run_ga_drl_comparison(drl_model_path, scenarios_to_compare, 
                          ga_generations=50, ga_pop_size=20, ga_mutation_percent=10):
    """
    Main function to run GA and DRL comparisons.
    """
    global ga_current_scenario_base_inp_path, ga_current_scenario_original_diameters
    global ga_current_scenario_max_pd, ga_simulation_calls_counter

    comparison_data = []
    
    for scenario_name in scenarios_to_compare:
        print(f"\n--- Starting Comparison for Scenario: {scenario_name} ---")

        # --- Setup for both GA and DRL final state evaluation ---
        scenario_path = os.path.join(NETWORKS_FOLDER_PATH, scenario_name)
        inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])
        if not inp_files:
            print(f"  No .inp files found for scenario {scenario_name}. Skipping.")
            continue
        
        # The GA optimizes the final network state of the scenario
        final_inp_filename = inp_files[-1]
        ga_current_scenario_base_inp_path = os.path.join(scenario_path, final_inp_filename)
        
        wn_base = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
        ga_current_scenario_original_diameters = {
            p_name: wn_base.get_link(p_name).diameter for p_name in wn_base.pipe_name_list
        }
        num_pipes = len(wn_base.pipe_name_list)
        num_nodes = len(wn_base.node_name_list)
        network_size_str = f"Nodes: {num_nodes}, Pipes: {num_pipes}"

        # Calculate max_pd for this base network (used by both GA fitness and DRL final eval)
        # This is a "setup" simulation call for the scenario.
        ga_current_scenario_max_pd, sim_calls_for_max_pd_setup = calculate_max_pd_for_scenario(ga_current_scenario_base_inp_path, PIPES_CONFIG)
        
        # --- GA Run ---
        print(f"  Running GA for {scenario_name} (target: {final_inp_filename})...")
        ga_simulation_calls_counter = 0 # Reset for this specific GA run
        start_time_ga = time.time()

        ga_instance = pygad.GA(
            num_generations=ga_generations,
            num_parents_mating=max(2, int(ga_pop_size * 0.2)), # Ensure at least 2 parents
            fitness_func=ga_fitness_function,
            sol_per_pop=ga_pop_size,
            num_genes=num_pipes,
            gene_type=int,
            gene_space={'low': 0, 'high': len(PIPE_DIAMETER_OPTIONS) - 1},
            mutation_percent_genes=ga_mutation_percent,
            # on_generation=lambda ga_inst: print(f"  GA Gen: {ga_inst.generations_completed}, Best Fitness: {ga_inst.best_solution()[1]:.2f}")
        )
        ga_instance.run()
        ga_run_time = time.time() - start_time_ga
        
        # Total GA simulation calls = calls during run + 1 for max_pd setup
        total_ga_sim_calls = ga_simulation_calls_counter + sim_calls_for_max_pd_setup
        
        best_ga_solution_chromosome, best_ga_fitness, _ = ga_instance.best_solution()
        
        # Evaluate the best GA solution to get detailed metrics
        wn_best_ga = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
        ga_actions_for_best = []
        for i, pipe_name in enumerate(wn_best_ga.pipe_name_list):
            chosen_idx = int(best_ga_solution_chromosome[i])
            new_diam = PIPE_DIAMETER_OPTIONS[chosen_idx]
            if abs(new_diam - ga_current_scenario_original_diameters[pipe_name]) > 1e-6:
                ga_actions_for_best.append((pipe_name, new_diam))
            wn_best_ga.get_link(pipe_name).diameter = new_diam
        
        # This is an additional simulation to get metrics for the GA's best found solution
        # It's fair to count it as part of GA's overhead if DRL's final eval sim is also counted.
        total_ga_sim_calls += 1 
        ga_best_results, ga_best_metrics = run_epanet_simulation(wn_best_ga)
        
        ga_cost, ga_pd_abs, ga_demand_sat_ratio = float('inf'), float('inf'), 0
        if ga_best_results and not ga_best_results.node['pressure'].isnull().values.any():
            # Recalculate cost based on actions from original .inp
            ga_cost = compute_total_cost(
                initial_pipes=wn_best_ga.pipes(), # Not strictly needed, actions list is key
                actions=ga_actions_for_best,
                labour_cost=LABOUR_COST,
                energy_cost=ga_best_metrics['total_pump_cost'],
                pipes=PIPES_CONFIG,
                original_pipe_diameters=ga_current_scenario_original_diameters
            )
            ga_pd_abs = ga_best_metrics.get('total_pressure_deficit', float('inf'))
            ga_demand_sat_ratio = ga_best_metrics.get('demand_satisfaction_ratio', 0)
        
        comparison_data.append({
            'Scenario': scenario_name, 'Method': 'GA', 'Network Size': network_size_str,
            'Fitness/Reward': best_ga_fitness, 'Total Cost (£)': ga_cost,
            'Pressure Deficit (m)': ga_pd_abs, 'Demand Satisfaction (%)': ga_demand_sat_ratio * 100,
            'Simulation Calls': total_ga_sim_calls, 'Time (s)': ga_run_time
        })
        print(f"  GA Results - Fitness: {best_ga_fitness:.2f}, Cost: {ga_cost:.2f}, PD: {ga_pd_abs:.2f}, Demand Sat: {ga_demand_sat_ratio*100:.2f}%, Sim Calls: {total_ga_sim_calls}, Time: {ga_run_time:.2f}s")

        # --- DRL Evaluation ---
        print(f"  Evaluating DRL agent on {scenario_name} for final state comparison...")
        start_time_drl = time.time()
        drl_metrics = evaluate_drl_agent_on_final_state(drl_model_path, scenario_name, PIPES_CONFIG, LABOUR_COST)
        drl_run_time = time.time() - start_time_drl

        comparison_data.append({
            'Scenario': scenario_name, 'Method': 'DRL', 'Network Size': network_size_str,
            'Fitness/Reward': drl_metrics['reward'], 'Total Cost (£)': drl_metrics['cost'],
            'Pressure Deficit (m)': drl_metrics['pd'], 'Demand Satisfaction (%)': drl_metrics['demand_sat'] * 100,
            'Simulation Calls': drl_metrics['sim_calls'], 'Time (s)': drl_run_time
        })
        print(f"  DRL Results - Reward: {drl_metrics['reward']:.2f}, Cost: {drl_metrics['cost']:.2f}, PD: {drl_metrics['pd']:.2f}, Demand Sat: {drl_metrics['demand_sat']*100:.2f}%, Sim Calls: {drl_metrics['sim_calls']}, Time: {drl_run_time:.2f}s")

    return pd.DataFrame(comparison_data)


def plot_comparison_results(df_results, timestamp):
    """
    Generates and saves plots comparing GA and DRL results.
    """
    plots_dir = os.path.join("Plots", "GA_DRL_Comparison_Charts")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_to_plot = {
        'Total Cost (£)': 'Total Cost (£): GA vs DRL',
        'Pressure Deficit (m)': 'Pressure Deficit (m): GA vs DRL',
        'Demand Satisfaction (%)': 'Demand Satisfaction (%): GA vs DRL',
        'Simulation Calls': 'Computational Overhead (Simulation Calls): GA vs DRL',
        'Time (s)': 'Computational Time (s): GA vs DRL'
    }
    
    scenarios = df_results['Scenario'].unique()
    num_scenarios = len(scenarios)
    x = np.arange(num_scenarios)  # the label locations
    width = 0.35  # the width of the bars

    for metric_col, title in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(max(10, num_scenarios * 2.5 + 2), 7)) # Dynamic width
        
        ga_values = df_results[df_results['Method'] == 'GA'][metric_col].values
        drl_values = df_results[df_results['Method'] == 'DRL'][metric_col].values
        
        # Ensure arrays match length of scenarios if a method failed for one
        ga_plot_vals = [df_results[(df_results['Method'] == 'GA') & (df_results['Scenario'] == s)][metric_col].iloc[0] if not df_results[(df_results['Method'] == 'GA') & (df_results['Scenario'] == s)].empty else 0 for s in scenarios]
        drl_plot_vals = [df_results[(df_results['Method'] == 'DRL') & (df_results['Scenario'] == s)][metric_col].iloc[0] if not df_results[(df_results['Method'] == 'DRL') & (df_results['Scenario'] == s)].empty else 0 for s in scenarios]


        rects1 = ax.bar(x - width/2, ga_plot_vals, width, label='GA')
        rects2 = ax.bar(x + width/2, drl_plot_vals, width, label='DRL')

        ax.set_ylabel(metric_col.split('(')[0].strip()) # Cleaner Y-axis label
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha="right")
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        fig.tight_layout()
        plot_filename = os.path.join(plots_dir, f"comparison_{metric_col.replace(' (£)', '').replace(' (m)', '').replace(' (%)', '').replace(' ', '_').lower()}_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Saved comparison plot: {plot_filename}")
        plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    # --- Configuration for the comparison ---
    # Ensure you have a trained DRL model. Replace with your actual model path.
    # Example: latest_drl_model_path = "agents/trained_gnn_ppo_wn_YYYYMMDD_HHMMSS.zip" 
    # For testing, let's assume a model path. You'll need to provide a real one.
    
    # Find the latest trained DRL model in the "agents" directory
    agents_dir = "agents"
    if not os.path.exists(agents_dir):
        print(f"Error: Agents directory '{agents_dir}' not found. Please train a DRL model first or specify a path.")
        exit()

    list_of_files = [os.path.join(agents_dir, f) for f in os.listdir(agents_dir) if f.startswith("trained_gnn_ppo_wn_") and f.endswith(".zip")]
    if not list_of_files:
        print(f"Error: No trained DRL models found in '{agents_dir}'. Please train a DRL model first.")
        exit()
    
    latest_drl_model_path = max(list_of_files, key=os.path.getctime)
    print(f"Using latest DRL model for comparison: {latest_drl_model_path}")

    scenarios_for_comparison = ['hanoi_sprawling_3', 'anytown_sprawling_3']
    
    # GA Parameters (can be tuned)
    ga_generations = 50  # Number of generations
    ga_population_size = 30 # Population size
    ga_mutation_rate = 15 # Mutation rate (percentage of genes)

    # --- Run the comparison ---
    print("Starting GA vs DRL Comparison...")
    start_total_comparison_time = time.time()
    
    comparison_df = run_ga_drl_comparison(
        drl_model_path=latest_drl_model_path,
        scenarios_to_compare=scenarios_for_comparison,
        ga_generations=ga_generations,
        ga_pop_size=ga_population_size,
        ga_mutation_percent=ga_mutation_rate
    )
    
    total_comparison_time = time.time() - start_total_comparison_time
    print(f"\nComparison run completed in {total_comparison_time:.2f} seconds.")

    # --- Print and Plot Results ---
    print("\n--- Comparison Results ---")
    print(comparison_df.to_string())

    # Save results to CSV
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"ga_drl_comparison_results_{timestamp_str}.csv")
    comparison_df.to_csv(csv_filename, index=False)
    print(f"\nComparison results saved to: {csv_filename}")

    print("\nGenerating comparison plots...")
    plot_comparison_results(comparison_df, timestamp_str)
    
    print("\nAll comparison tasks complete!")
    plt.show() # Show all plots at the very end if desired, or rely on saved files.

if __name__ == "__main__":

    # Test the GA scipt on a single scenario

    test_scenario = 'anytown_sprawling_3' 