
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
from Hydraulic_2 import run_epanet_simulation, evaluate_network_performance
from Reward2 import calculate_reward, compute_total_cost
from PPO_Environment2 import WNTRGymEnv # For DRL evaluation
from Actor_Critic_Nets3 import GraphPPOAgent # For DRL evaluation

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
ENERGY_COST = 0.26  # Cost per kWh, ensure this matches the value in WNTRGymEnv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORKS_FOLDER_PATH = os.path.join(SCRIPT_DIR, 'Networks2')
EXCLUDE_PIPES_ANYTOWN = []
EXCLUDE_PIPES_HANOI = []

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
    
def calculate_max_cost_for_scenario(base_inp_path, pipes, labour_cost, energy_cost):

    wn = wntr.network.WaterNetworkModel(base_inp_path)
    initial_pipes = list(wn.pipes())

    pipe_ids = []
    pipe_diams = []
    for pipe, pipe_data in initial_pipes:
        pipe_ids.append(pipe)
        pipe_diams.append(pipe_data.diameter)
    
    # Create a dictionary of original pipe diameters
    original_pipe_diameters = dict(zip(pipe_ids, pipe_diams))

    # original_pipe_diameters = {pipe.name: pipe.diameter for pipe in initial_pipes}

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
        # Get the current diameter from original_pipe_diameters if available
        if pipe_id in original_pipe_diameters:
            current_diameter = original_pipe_diameters[pipe_id]
        else:
            # Otherwise get it from the current network
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

    print("-------------------------------------")
    print("Calculating cost given maximum actions...")

    max_cost = compute_total_cost(initial_pipes, max_actions, labour_cost, energy_cost, pipes, original_pipe_diameters)

    return max_cost

# In GA_Alt_Approach2.py

from Reward2 import _reward_custom_normalized # Import the specific function

def ga_fitness_function(ga_instance, solution, solution_idx):
    """
    (Corrected) Fitness function for the Genetic Algorithm.
    Uses the same reward calculation as the DRL environment.
    """
    global ga_simulation_calls_counter, ga_current_scenario_pipes_to_optimize, ga_current_scenario_original_diameters
    global ga_current_scenario_max_pd, ga_current_scenario_max_cost, ga_baseline_metrics
    
    ga_simulation_calls_counter += 1

    wn_trial = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
    
    actions_for_this_solution = []
    
    for i, pipe_name in enumerate(ga_current_scenario_pipes_to_optimize):
        chosen_diameter_index = int(solution[i])
        new_diameter = PIPE_DIAMETER_OPTIONS[chosen_diameter_index]
        
        # Prevent downgrading, which is a constraint on the DRL agent
        if new_diameter < ga_current_scenario_original_diameters[pipe_name]:
            return -1e9 # Penalize solutions that downgrade pipes

        wn_trial.get_link(pipe_name).diameter = new_diameter
        actions_for_this_solution.append((pipe_name, new_diameter))

    sim_results = run_epanet_simulation(wn_trial)
    if not sim_results or sim_results.node['pressure'].isnull().values.any():
        return -1e10

    sim_metrics = evaluate_network_performance(wn_trial, sim_results)
    
    cost_of_intervention, _, _, _ = compute_total_cost(
        actions=actions_for_this_solution,
        pipes_config=PIPES_CONFIG,
        wn=wn_trial,
        energy_cost=sim_metrics.get('total_pump_cost', 0),
        labour_cost_per_meter=LABOUR_COST
    )

    # --- Create the params dictionary to match the DRL environment's reward call ---
    reward_params = {
        'metrics': sim_metrics,
        'cost_of_intervention': cost_of_intervention,
        'baseline_pressure_deficit': ga_baseline_metrics['total_pressure_deficit'],
        'baseline_demand_satisfaction': ga_baseline_metrics['demand_satisfaction_ratio'],
        'current_budget': float('inf'), # Assume GA is not budget constrained for this single-step optimization
        'max_cost_normalization': 1000000.0 # From your Train_w_Plots2.py REWARD_CONFIG
    }
    
    # Use the same reward function as the DRL environment
    reward_val, _ = _reward_custom_normalized(params=reward_params)
    
    return float(reward_val)

# In GA_Alt_Approach2.py

def evaluate_drl_on_single_state(drl_model_path, base_inp_path, scenario_name, baseline_metrics):
    """
    (New) Evaluates the DRL agent's performance on a single network state.
    """
    print(f"  Evaluating DRL on: {os.path.basename(base_inp_path)}")
    
    # The environment needs a scenario list, but we force it to use our specific .inp file
    # by manipulating its internal state after reset.
    env_configs = {
        'pipes_config': PIPES_CONFIG, 'scenarios': [scenario_name], 
        'network_config': {'max_nodes': 150, 'max_pipes': 200},
        'budget_config': {"max_debt": float('inf')}, # Disable budget constraints for this test
        'reward_config': {'mode': 'custom_normalized', 'max_cost_normalization': 1000000.0}
    }
    env = WNTRGymEnv(**env_configs)
    
    # Manually set the environment to the specific state we want to test
    obs, info = env.reset(options={'scenario_name': scenario_name})
    env.current_network = wntr.network.WaterNetworkModel(base_inp_path)
    env.pipe_names = env.current_network.pipe_name_list
    env.baseline_pressure_deficit = baseline_metrics['total_pressure_deficit']
    env.baseline_demand_satisfaction = baseline_metrics['demand_satisfaction_ratio']
    obs = env._get_network_features() # Regenerate observation with the correct network

    agent = GraphPPOAgent(env, pipes_config=PIPES_CONFIG)
    agent.load(drl_model_path)
    
    # Let the DRL agent make decisions for every pipe in this single state
    for _ in env.pipe_names:
        action_masks = env.action_masks()
        action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, _, _, _ = env.step(action.item())

    # The loop is done, env.current_network now holds the DRL's final configuration
    final_drl_network = env.current_network
    
    # Now, evaluate this final configuration
    final_sim_results = run_epanet_simulation(final_drl_network)
    if not final_sim_results:
        return {'reward': -1e10, 'cost': float('inf')}
        
    final_metrics = evaluate_network_performance(final_drl_network, final_sim_results)
    
    # Identify actions taken by DRL to calculate cost
    original_diameters = {p.name: p.diameter for _, p in wntr.network.WaterNetworkModel(base_inp_path).pipes()}
    actions_taken = []
    for pipe_name in final_drl_network.pipe_name_list:
        new_diam = final_drl_network.get_link(pipe_name).diameter
        if abs(new_diam - original_diameters.get(pipe_name, new_diam)) > 1e-6:
            actions_taken.append((pipe_name, new_diam))

    cost_of_intervention, _, _, _ = compute_total_cost(
        actions=actions_taken, pipes_config=PIPES_CONFIG, wn=final_drl_network,
        energy_cost=final_metrics.get('total_pump_cost', 0), labour_cost_per_meter=LABOUR_COST
    )

    reward_params = {
        'metrics': final_metrics, 'cost_of_intervention': cost_of_intervention,
        'baseline_pressure_deficit': baseline_metrics['total_pressure_deficit'],
        'baseline_demand_satisfaction': baseline_metrics['demand_satisfaction_ratio'],
        'current_budget': float('inf'), 'max_cost_normalization': 1000000.0
    }
    reward_val, _ = _reward_custom_normalized(params=reward_params)
    
    env.close()
    return {'reward': reward_val, 'cost': cost_of_intervention}

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
    
    # final_inp_file_for_scenario = inp_files[-1] # Use first file instead!
    final_inp_file_for_scenario = 'Step_50.inp'

    final_inp_path = os.path.join(scenario_path, final_inp_file_for_scenario)

    wn_final_base = wntr.network.WaterNetworkModel(final_inp_path)
    
    original_diameters_of_final_inp = {}
    for p_name, p_data in wn_final_base.pipes():
        original_diameters_of_final_inp[p_data.name] = p_data.diameter
    
    print(f"  Original diameters for final .inp file {final_inp_file_for_scenario}: {original_diameters_of_final_inp}")
    
    max_pd_for_final_inp, sim_calls_max_pd = calculate_max_pd_for_scenario(final_inp_path, pipes_config_dict)
    drl_sim_calls_for_this_eval += sim_calls_max_pd

    max_cost = calculate_max_cost_for_scenario(
        ga_current_scenario_base_inp_path, 
        pipes=PIPES_CONFIG, 
        labour_cost=LABOUR_COST, 
        energy_cost=ENERGY_COST
    )

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
            # for p_name in final_drl_network_configuration.pipe_name_list:
            for p_name in final_drl_network_configuration.pipe_name_list:
                drl_chosen_diameter = final_drl_network_configuration.get_link(p_name).diameter

                print(f"  DRL chose diameter {drl_chosen_diameter} for pipe {p_name} in final step.")

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
    final_drl_results = run_epanet_simulation(final_drl_network_configuration)
    final_drl_metrics = evaluate_network_performance(final_drl_network_configuration, final_drl_results)
    
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
        max_pd=max_pd_for_final_inp,
        max_cost=max_cost,
    )
    
    drl_abs_pd = final_drl_metrics.get('total_pressure_deficit', float('inf'))
    
    return {
        'reward': drl_reward,
        'cost': drl_cost,
        'pd': drl_abs_pd,
        'demand_sat': drl_demand_sat, # This is already a ratio
        'sim_calls': drl_sim_calls_for_this_eval,
        'network_size': (len(final_drl_network_configuration.node_name_list), len(final_drl_network_configuration.pipe_name_list)),
        'network': final_drl_network_configuration, # Return the network for further analysis if needed
    }

def run_single_scenario_comparison(
    drl_model_path: str, 
    target_scenario_name: str,
    pipes_config_dict: dict,
    labour_cost_val: float,
    ga_generations: int = 50, 
    ga_pop_size: int = 20, 
    ga_mutation_percent: int = 10,
    ga_crossover_probability: float = 0.8,
    ga_keep_elitism: int = 2
):
    global ga_current_scenario_base_inp_path, ga_current_scenario_original_diameters
    global ga_current_scenario_max_pd, ga_current_scenario_max_cost
    global ga_simulation_calls_counter, ga_current_scenario_pipes_to_optimize

    comparison_results = {} # Store results for this single comparison

    print(f"\n--- Starting Comparison for Scenario: {target_scenario_name} ---")

    # --- Setup for both GA and DRL final state evaluation ---
    scenario_path = os.path.join(NETWORKS_FOLDER_PATH, target_scenario_name)
    inp_files_list = [f for f in os.listdir(scenario_path) if f.endswith('.inp')]
    if not inp_files_list:
        print(f"  No .inp files found for scenario {target_scenario_name}. Skipping.")
        return None
    try: # Sort .inp files by step number
        inp_files_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    except Exception as e:
        print(f"Could not sort .inp files numerically for {target_scenario_name}, using lexicographical sort. Error: {e}")
        inp_files_list.sort()

    # GA optimizes the final network state of the scenario
    final_inp_filename = inp_files_list[-1]

    print(f"  Final .inp file for GA optimization: {final_inp_filename}")

    ga_current_scenario_base_inp_path = os.path.join(scenario_path, final_inp_filename)
    
    wn_base_for_ga_and_final_drl = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
    ga_current_scenario_original_diameters = {
        p_name: wn_base_for_ga_and_final_drl.get_link(p_name).diameter 
        for p_name in wn_base_for_ga_and_final_drl.pipe_name_list
    }
    num_pipes = len(wn_base_for_ga_and_final_drl.pipe_name_list)

    # Calculate max_pd for this final network state (used by GA fitness and DRL final eval)
    ga_current_scenario_max_pd, sim_calls_for_max_pd_setup = calculate_max_pd_for_scenario(
        ga_current_scenario_base_inp_path, pipes_config_dict
    )

    # calculate max_cost
    ga_current_scenario_max_cost = calculate_max_cost_for_scenario(
        ga_current_scenario_base_inp_path, pipes = PIPES_CONFIG, labour_cost = LABOUR_COST, energy_cost = ENERGY_COST)
    
    print(f"  Max PD for {final_inp_filename}: {ga_current_scenario_max_pd}, Max Cost: {ga_current_scenario_max_cost}")
    
    # --- GA Run ---
    print(f"  Running GA for {target_scenario_name} (optimizing state of: {final_inp_filename})...")
    ga_simulation_calls_counter = 0 
    start_time_ga = time.time()

    # Check the network type and exclude the appropriate pipes
    # if target_scenario_name includes 'Anytown':
    if 'Anytown' in target_scenario_name:
        excluded_pipes = EXCLUDE_PIPES_ANYTOWN
    elif 'Hanoi' in target_scenario_name:
        excluded_pipes = EXCLUDE_PIPES_HANOI
    else:
        excluded_pipes = []

    # Get list of pipes to optimize (exclude the specified pipes)
    ga_current_scenario_pipes_to_optimize = [
        p_name for p_name in wn_base_for_ga_and_final_drl.pipe_name_list 
        if p_name not in excluded_pipes
    ]
    
    # Set number of genes based on pipes to optimize
    num_genes = len(ga_current_scenario_pipes_to_optimize)
    
    print(f"  Optimizing {num_genes} pipes (excluded {len(excluded_pipes)} pipes: {excluded_pipes})")
    
    # Initialize GA instance with only the pipes we want to optimize
    ga_instance = pygad.GA(
        num_generations=ga_generations,
        num_parents_mating=max(2, int(ga_pop_size * 0.2)),
        fitness_func=ga_fitness_function,
        sol_per_pop=ga_pop_size,
        num_genes=num_genes,  # Only optimize non-excluded pipes
        gene_type=int,
        gene_space={'low': 0, 'high': len(PIPE_DIAMETER_OPTIONS) - 1},
        mutation_percent_genes=ga_mutation_percent,
        crossover_probability=ga_crossover_probability,
        keep_elitism=ga_keep_elitism,
        parent_selection_type="tournament",
        K_tournament=3,
        stop_criteria=["reach_1.0", "saturate_10"]
    )
    
    ga_instance.run()
    ga_run_time = time.time() - start_time_ga # This is GA's "training time"
    
    total_ga_sim_calls_during_opt = ga_simulation_calls_counter
    
    # When processing best solution, make sure to match the chromosome to the right pipes
    best_ga_solution_chromosome, best_ga_fitness, _ = ga_instance.best_solution()
    
    # Construct the GA's best network and identify upgrades
    wn_best_ga = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
    ga_proposed_upgrades = []
    ga_actions_for_reward_calc = []

    for i, pipe_name in enumerate(ga_current_scenario_pipes_to_optimize):
        chosen_idx = int(best_ga_solution_chromosome[i])
        new_diam = PIPE_DIAMETER_OPTIONS[chosen_idx]
        original_diam = ga_current_scenario_original_diameters[pipe_name]
        
        wn_best_ga.get_link(pipe_name).diameter = new_diam
        if abs(new_diam - original_diam) > 1e-6:
            ga_proposed_upgrades.append((pipe_name, original_diam, new_diam))
            ga_actions_for_reward_calc.append((pipe_name, new_diam))

    # Evaluate the best GA solution to get detailed metrics (this is one final simulation)
    ga_final_sim_results = run_epanet_simulation(wn_best_ga)
    ga_final_sim_metrics = evaluate_network_performance(wn_best_ga, ga_final_sim_results)
    
    ga_reward_final, ga_cost_final, ga_pd_ratio_final, ga_demand_sat_final, _, _, _ = calculate_reward(
        current_network=wn_best_ga,
        original_pipe_diameters=ga_current_scenario_original_diameters, # Compare to original final step
        actions=ga_actions_for_reward_calc, # Actions GA took relative to original final step
        pipes=pipes_config_dict,
        performance_metrics=ga_final_sim_metrics,
        labour_cost=labour_cost_val,
        downgraded_pipes=False, # GA should learn not to downgrade if fitness is penalized
        max_pd=ga_current_scenario_max_pd, # Use max_pd of the base final state
        max_cost=ga_current_scenario_max_cost) # Use max_cost of the base final state)
    
    comparison_results['GA'] = {
        'Method': 'GA',
        'Scenario': target_scenario_name,
        'Target_File': final_inp_filename,
        'Training_Time_s': ga_run_time,
        'Sim_Calls_Optimization': total_ga_sim_calls_during_opt,
        'Sim_Calls_Setup_Eval': sim_calls_for_max_pd_setup + 1, # +1 for final solution eval
        'Best_Fitness_Raw': best_ga_fitness, # This is raw from fitness_func
        'Final_Calculated_Reward': ga_reward_final,
        'Final_Cost': ga_cost_final,
        'Proposed_Upgrades': ga_proposed_upgrades
    }
    print(f"  GA Best Solution for {final_inp_filename} - Calculated Reward: {ga_reward_final:.2f}, Cost: {ga_cost_final:.2f}")
    print(f"  GA Proposed Upgrades ({len(ga_proposed_upgrades)}): {ga_proposed_upgrades}")

    # --- DRL Evaluation: Time for one full episode ---
    print(f"  Evaluating DRL agent on full episode of {target_scenario_name} for timing and final state...")
    start_time_drl_episode = time.time()
    
    # evaluate_drl_agent_on_final_state runs the full episode and evaluates the *final* configuration
    # We are interested in the *time taken to run one full episode* for the DRL agent
    # The existing function already measures the time for DRL agent to process one scenario to get final state
    
    # We need a separate function or adaptation to get the time for ONE DRL episode run
    # For now, let's use evaluate_drl_agent_on_final_state and report its DRL run time as "evaluation time"
    # and its reported "reward" as the reward for its final configuration.
    
    drl_final_state_metrics = evaluate_drl_agent_on_final_state(
        drl_model_path, target_scenario_name, pipes_config_dict, labour_cost_val
    )
    drl_episode_run_time = time.time() - start_time_drl_episode # This is the time for the DRL eval function to run

    comparison_results['DRL'] = {
        'Method': 'DRL',
        'Scenario': target_scenario_name,
        'Target_File': final_inp_filename, # DRL also effectively targets this by end of episode
        'Episode_Run_Time_s': drl_episode_run_time, # Time to get to final state and evaluate it
        'Sim_Calls_Episode': drl_final_state_metrics.get('sim_calls', 'N/A'), # Total calls during the episode
        'Final_Calculated_Reward': drl_final_state_metrics.get('reward', 'N/A'),
        'Final_Cost': drl_final_state_metrics.get('cost', 'N/A'),
        'Proposed_Upgrades': "Inspect DRL agent's actions on final step via other means" # Not directly extracted here
    }
    print(f"  DRL Full Episode Evaluation for {target_scenario_name} - Time: {drl_episode_run_time:.2f}s, Final State Reward: {drl_final_state_metrics.get('reward', 'N/A'):.2f}")

    return comparison_results, ga_final_sim_metrics, drl_final_state_metrics

def create_pipe_diameter_comparison(ga_network, drl_network, original_network, scenario_name, timestamp):
    """
    Creates a DataFrame comparing pipe diameters from original network, GA solution, and DRL solution.
    Also saves the DataFrame to CSV and generates a visual comparison.
    
    Args:
        ga_network: The best network solution from GA
        drl_network: The final network configuration from DRL
        original_network: The original network before any modifications
        scenario_name: Name of the scenario being compared
        timestamp: Timestamp for file naming
    """
    # Get all pipe names (union of all networks to handle any differences)
    all_pipe_names = set(original_network.pipe_name_list)
    all_pipe_names.update(ga_network.pipe_name_list if ga_network else [])
    all_pipe_names.update(drl_network.pipe_name_list if drl_network else [])
    all_pipe_names = sorted(list(all_pipe_names))
    
    # Create dictionary to store diameter data
    diameter_data = {
        'pipe_name': all_pipe_names,
        'original_diameter': [],
        'ga_diameter': [],
        'drl_diameter': []
    }
    
    # Fill in diameter values
    for pipe_name in all_pipe_names:
        # Original diameter
        if pipe_name in original_network.pipe_name_list:
            diameter_data['original_diameter'].append(original_network.get_link(pipe_name).diameter)
        else:
            diameter_data['original_diameter'].append(None)
        
        # GA diameter
        if ga_network and pipe_name in ga_network.pipe_name_list:
            diameter_data['ga_diameter'].append(ga_network.get_link(pipe_name).diameter)
        else:
            diameter_data['ga_diameter'].append(None)
        
        # DRL diameter
        if drl_network and pipe_name in drl_network.pipe_name_list:
            diameter_data['drl_diameter'].append(drl_network.get_link(pipe_name).diameter)
        else:
            diameter_data['drl_diameter'].append(None)
    
    # Create DataFrame
    diameter_df = pd.DataFrame(diameter_data)
    
    # Calculate changes from original
    diameter_df['ga_change'] = diameter_df['ga_diameter'] - diameter_df['original_diameter']
    diameter_df['drl_change'] = diameter_df['drl_diameter'] - diameter_df['original_diameter']
    
    # Save to CSV
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"pipe_diameters_{scenario_name.replace('/', '_')}_{timestamp}.csv")
    diameter_df.to_csv(csv_filename, index=False)
    print(f"Saved pipe diameter comparison to: {csv_filename}")
    
    # Create visualization for a subset of pipes (top 10 with largest changes)
    plot_pipe_diameter_comparison(diameter_df, scenario_name, timestamp)
    
    return diameter_df

def plot_pipe_diameter_comparison(diameter_df, scenario_name, timestamp):
    """Create a visual comparison of pipe diameters between GA and DRL solutions."""
    # Find pipes with the largest absolute changes (either GA or DRL)
    diameter_df['max_abs_change'] = diameter_df[['ga_change', 'drl_change']].abs().max(axis=1)
    
    # Get top 10 pipes with largest changes
    top_pipes = diameter_df.nlargest(10, 'max_abs_change')
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Set positions for grouped bars
    pipe_indices = np.arange(len(top_pipes))
    bar_width = 0.25
    
    # Plot bars for each solution
    plt.bar(pipe_indices - bar_width, top_pipes['original_diameter'], 
            width=bar_width, label='Original', alpha=0.7)
    plt.bar(pipe_indices, top_pipes['ga_diameter'], 
            width=bar_width, label='GA Solution')
    plt.bar(pipe_indices + bar_width, top_pipes['drl_diameter'], 
            width=bar_width, label='DRL Solution')
    
    # Add labels and title
    plt.xlabel('Pipe Name')
    plt.ylabel('Diameter (m)')
    plt.title(f'Pipe Diameter Comparison - Top 10 Changed Pipes\nScenario: {scenario_name}')
    plt.xticks(pipe_indices, top_pipes['pipe_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join("Plots", "GA_DRL_Comparison_Charts")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"diameter_comparison_{scenario_name.replace('/', '_')}_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved pipe diameter comparison plot: {plot_filename}")
    plt.close()

def plot_time_comparison(ga_results, drl_results, target_scenario_name, timestamp):
    """Plots the time comparison."""
    plt.figure(figsize=(8, 6))
    methods = ['GA Optimization Time', 'DRL Episode Run Time']
    times = [ga_results.get('Training_Time_s', 0), drl_results.get('Episode_Run_Time_s', 0)]
    
    bars = plt.bar(methods, times)
    plt.ylabel('Time (s)')
    plt.title(f'Time Comparison for Scenario: {target_scenario_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(times), f'{yval:.2f}s', ha='center', va='bottom')

    plots_dir = os.path.join("Plots", "GA_DRL_Comparison_Charts")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"time_comparison_{target_scenario_name.replace('/', '_')}_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved time comparison plot: {plot_filename}")
    plt.close()

def plot_reward_comparison(ga_results, drl_results, target_scenario_name, timestamp):
    """Plots the reward comparison."""
    plt.figure(figsize=(8, 6))
    methods = ['GA Best Solution', 'DRL Final Configuration']
    rewards = [ga_results.get('Final_Calculated_Reward', 0), drl_results.get('Final_Calculated_Reward', 0)]
    
    bars = plt.bar(methods, rewards)
    plt.ylabel('Reward')
    plt.title(f'Reward Comparison for Scenario: {target_scenario_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(rewards), f'{yval:.2f}', ha='center', va='bottom')

    plots_dir = os.path.join("Plots", "GA_DRL_Comparison_Charts")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"reward_comparison_{target_scenario_name.replace('/', '_')}_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved reward comparison plot: {plot_filename}")
    plt.close()

def run_ga_on_single_state(base_inp_path, baseline_metrics_for_ga):
    """(New) Runs the GA for a single network state."""
    global ga_current_scenario_base_inp_path, ga_current_scenario_original_diameters
    global ga_baseline_metrics, ga_simulation_calls_counter, ga_current_scenario_pipes_to_optimize

    print(f"  Running GA on: {os.path.basename(base_inp_path)}")
    ga_current_scenario_base_inp_path = base_inp_path
    ga_baseline_metrics = baseline_metrics_for_ga

    wn_base = wntr.network.WaterNetworkModel(base_inp_path)
    ga_current_scenario_original_diameters = {p.name: p.diameter for _, p in wn_base.pipes()}
    ga_current_scenario_pipes_to_optimize = wn_base.pipe_name_list
    
    ga_instance = pygad.GA(
        num_generations=50, # Keep GA runs shorter for the loop
        num_parents_mating=5,
        fitness_func=ga_fitness_function,
        sol_per_pop=20,
        num_genes=len(ga_current_scenario_pipes_to_optimize),
        gene_type=int,
        gene_space={'low': 0, 'high': len(PIPE_DIAMETER_OPTIONS) - 1},
        mutation_percent_genes=10,
        stop_criteria="saturate_5"
    )
    ga_instance.run()
    
    best_solution, best_fitness, _ = ga_instance.best_solution()
    return {'reward': best_fitness} # Fitness is already the reward

if __name__ == "__main__":
    # --- SETUP ---
    script = os.path.dirname(os.path.abspath(__file__))
    drl_model_path = os.path.join(script, "agents", "Anytown_Only_20250618_190616.zip") # <--- UPDATE YOUR DRL MODEL PATH
    
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
        'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
        'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]

    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    
    results_list = []

    # --- MAIN LOOP ---
    for scenario in anytown_scenarios: # Modify based on network type
        print(f"\n--- Processing Scenario: {scenario} ---")
        
        # Find the first .inp file for the scenario
        scenario_path = os.path.join(NETWORKS_FOLDER_PATH, scenario)
        try:
            inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if not inp_files: continue
            first_inp_path = os.path.join(scenario_path, inp_files[0])
        except (ValueError, IndexError):
            print(f"Could not find or sort .inp files for {scenario}. Skipping.")
            continue
            
        # 1. Calculate baseline metrics for the initial state (used by both GA and DRL)
        base_wn = wntr.network.WaterNetworkModel(first_inp_path)
        base_results = run_epanet_simulation(base_wn)
        if not base_results: continue
        baseline_metrics = evaluate_network_performance(base_wn, base_results)

        # 2. Run GA
        ga_results = run_ga_on_single_state(first_inp_path, baseline_metrics)
        
        # 3. Run DRL
        drl_results = evaluate_drl_on_single_state(drl_model_path, first_inp_path, scenario, baseline_metrics)
        
        # 4. Store results
        results_list.append({
            'Scenario': scenario,
            'GA_Reward': ga_results['reward'],
            'DRL_Reward': drl_results['reward']
        })

    # --- PLOTTING ---
    results_df = pd.DataFrame(results_list)
    print("\n--- Final Comparison Results ---")
    print(results_df)

    # Save results to CSV
    results_df.to_csv('GA_vs_DRL_First_Step_Comparison.csv', index=False)
    print("\nFull results saved to GA_vs_DRL_First_Step_Comparison.csv")
    
    # Create the comparison plot
    results_df.set_index('Scenario').plot(kind='bar', figsize=(15, 8), grid=True)
    plt.title('GA vs. DRL Performance on First Step of Each Scenario')
    plt.ylabel('Calculated Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('GA_vs_DRL_Comparison_Plot.png')
    plt.show()

# if __name__ == "__main__":

#     script = os.path.dirname(os.path.abspath(__file__))
#     agents_dir = os.path.join(script, "agents")
#     agent = os.path.join(agents_dir, "agent1_hanoi_only_20250605_115528")  # Adjust to your DRL model path

#     # drl_results = evaluate_drl_agent_on_final_state(
#     #     drl_model_path=agent,
#     #     target_scenario_name="hanoi_sprawling_3",
#     #     pipes_config_dict=PIPES_CONFIG,
#     #     labour_cost_val=LABOUR_COST
#     # )

#     # --- Scenario to Compare ---
#     # You specified training on Hanoi, so let's pick a Hanoi sprawling scenario
#     scenario_to_run_comparison = 'hanoi_sprawling_3' # Example

#     # GA Parameters
#     ga_generations = 2000  # Can be increased for better GA performance, but takes longer
#     ga_population_size = 30 # Encourage exploration
#     ga_mutation_rate = 15

#     print(f"\nStarting Single Scenario Comparison: DRL vs GA")
#     print(f"DRL Model: {agent}")
#     print(f"Scenario: {scenario_to_run_comparison}")
#     print(f"GA Params: Generations={ga_generations}, Pop_Size={ga_population_size}, Mut_Rate={ga_mutation_rate}%")

#     results, ga_final_sim_metrics, drl_final_sim_metrics = run_single_scenario_comparison(
#         drl_model_path=agent,
#         target_scenario_name=scenario_to_run_comparison,
#         pipes_config_dict=PIPES_CONFIG, #
#         labour_cost_val=LABOUR_COST, #
#         ga_generations=ga_generations,
#         ga_pop_size=ga_population_size,
#         ga_mutation_percent=ga_mutation_rate
#     )

#     if results:
#         print("\n--- Single Scenario Comparison Summary ---")
#         print(f"Scenario: {scenario_to_run_comparison}")
        
#         ga_res = results.get('GA', {})
#         drl_res = results.get('DRL', {})

#         print("\nGenetic Algorithm Results:")
#         print(f"  Target File for Optimization: {ga_res.get('Target_File')}")
#         print(f"  Optimization Time: {ga_res.get('Training_Time_s', 0):.2f} s")
#         print(f"  Sim Calls (Optimization): {ga_res.get('Sim_Calls_Optimization', 0)}")
#         print(f"  Sim Calls (Setup & Final Eval): {ga_res.get('Sim_Calls_Setup_Eval', 0)}")
#         print(f"  Best Solution Calculated Reward: {ga_res.get('Final_Calculated_Reward', 'N/A'):.4f}")
#         print(f"  Best Solution Cost: £{ga_res.get('Final_Cost', 'N/A'):.2f}")
#         print(f"  Proposed Upgrades ({len(ga_res.get('Proposed_Upgrades', []))} pipes changed):")
#         for pipe_id, orig_d, new_d in ga_res.get('Proposed_Upgrades', []):
#             print(f"    - Pipe {pipe_id}: {orig_d:.4f}m -> {new_d:.4f}m")

#         print("\nDRL Agent Results:")
#         print(f"  Full Episode Run Time: {drl_res.get('Episode_Run_Time_s', 0):.2f} s")
#         print(f"  Sim Calls (Full Episode): {drl_res.get('Sim_Calls_Episode', 'N/A')}")
#         print(f"  Final Configuration Reward: {drl_res.get('Final_Calculated_Reward', 'N/A'):.4f}")
#         print(f"  Final Configuration Cost: £{drl_res.get('Final_Cost', 'N/A'):.2f}")
#         print(f"  (DRL proposed upgrades for the final step are implicitly in its final network state)")

#         current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         plot_time_comparison(ga_res, drl_res, scenario_to_run_comparison, current_timestamp)
        
#         # Save detailed results to a file
#         results_df_single = pd.DataFrame([ga_res, drl_res])
#         results_dir = "Results"
#         os.makedirs(results_dir, exist_ok=True)
#         csv_filename_single = os.path.join(results_dir, f"single_comp_{scenario_to_run_comparison}_{current_timestamp}.csv")
#         results_df_single.to_csv(csv_filename_single, index=False)
#         print(f"\nDetailed comparison results saved to: {csv_filename_single}")

#         current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         plot_time_comparison(ga_res, drl_res, scenario_to_run_comparison, current_timestamp)
#         plot_reward_comparison(ga_res, drl_res, scenario_to_run_comparison, current_timestamp)

#         # Get the original network for comparison
#         original_network = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
        
#         # Reconstruct the GA's best network
#         ga_best_network = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
#         for pipe_id, orig_d, new_d in ga_res.get('Proposed_Upgrades', []):
#             ga_best_network.get_link(pipe_id).diameter = new_d

#         drl_final_network = drl_final_sim_metrics.get('network', None)
    
#         if drl_final_network is None:
#             # Fall back to creating a reconstructed network if needed
#             print("Note: DRL final network not available in metrics, reconstructing...")
#             drl_final_network = wntr.network.WaterNetworkModel(ga_current_scenario_base_inp_path)
            
#             # Apply any known actions if available
#             # This is left empty as your current code doesn't have a way to extract these actions
        
#         # Create and save the pipe diameter comparison
#         pipe_diameter_df = create_pipe_diameter_comparison(
#             ga_best_network, 
#             drl_final_network,
#             original_network,
#             scenario_to_run_comparison,
#             current_timestamp
#         )

#     print("\n--- Comparison Script Finished ---")