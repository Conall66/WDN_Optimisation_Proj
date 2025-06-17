
"""

This is the main file, in which an agent is trained,deployed and tested against the GA alternative.

"""



import numpy as np
import torch
import time
import datetime
import multiprocessing as mp
import os
import pandas as pd
import matplotlib.pyplot as plt
import wntr # Ensure wntr is imported for GA part
import pygad # Ensure pygad is imported for GA part

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback # For potential custom callbacks

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance # For GA and DRL eval
from Reward import calculate_reward, compute_total_cost # For GA and DRL eval

# --- Constants for GA and DRL Evaluation (shared) ---
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}
PIPE_DIAMETER_OPTIONS = sorted([p['diameter'] for p in PIPES_CONFIG.values()])
LABOUR_COST = 100
NETWORKS_FOLDER = 'Modified_nets' # Make sure this path is correct

# --- Global variables for GA fitness context (simplifies PyGAD integration) ---
# These will be set before each GA run for a specific timestep .inp file
ga_current_inp_path_for_fitness = None
ga_original_diameters_for_fitness = None
ga_max_pd_for_fitness = None
ga_simulation_calls_counter_for_fitness = 0


class MinimalLoggingCallback(BaseCallback):
    """
    A minimal callback for logging training progress.
    """
    def __init__(self, verbose=0):
        super(MinimalLoggingCallback, self).__init__(verbose)
        self.log_data = []
        self.last_log_time = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.last_log_time > 60: # Log every 60 seconds
            if self.logger:
                reward_mean = np.mean(self.logger.name_to_value.get('rollout/ep_rew_mean', 0))
                print(f"Timestep: {self.num_timesteps}, Mean Reward: {reward_mean:.2f}")
            self.last_log_time = time.time()
        return True

def train_drl_agent(training_scenarios: list, total_timesteps: int, model_save_name: str = "drl_agent_trained"):
    """
    Trains a DRL agent on the specified scenarios.
    """
    print(f"\n--- Training DRL Agent on {len(training_scenarios)} scenarios ---")
    print(f"Scenarios: {training_scenarios}")
    print(f"Total timesteps: {total_timesteps}")

    def make_env():
        # The WNTRGymEnv will internally cycle through the provided training_scenarios
        env = WNTRGymEnv(pipes=PIPES_CONFIG, scenarios=training_scenarios, networks_folder=NETWORKS_FOLDER)
        return env

    num_cpu = mp.cpu_count()
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 1
    }

    agent = GraphPPOAgent(vec_env, PIPES_CONFIG, **ppo_config)
    
    callback = MinimalLoggingCallback()

    start_time = time.time()
    agent.train(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time
    print(f"DRL Training completed in {training_time:.2f} seconds.")

    os.makedirs("agents", exist_ok=True)
    model_path = os.path.join("agents", model_save_name)
    agent.save(model_path)
    print(f"DRL Model saved to {model_path}.zip")
    
    vec_env.close()
    return model_path + ".zip" # SB3 automatically adds .zip

def deploy_drl_agent_on_scenario(trained_model_path: str, target_scenario_name: str):
    """
    Deploys the trained DRL agent on each timestep of a single target scenario.
    Collects performance metrics and computational time.
    """
    print(f"\n--- Deploying DRL Agent on Scenario: {target_scenario_name} ---")
    
    # Create a non-vectorized environment for controlled step-by-step evaluation
    # It's crucial that this env is initialized with *only* the target_scenario_name
    # so its internal reset logic for scenario cycling doesn't interfere.
    # However, WNTRGymEnv's reset can take a scenario_name, which is better.
    # We will use a DummyVecEnv as the agent expects a VecEnv interface,
    # but we will control the scenario explicitly.

    # This env will be used to step through the scenario's .inp files
    # The WNTRGymEnv's internal current_time_step will correspond to the .inp file index
    eval_env_list = [target_scenario_name] # WNTRGymEnv expects a list of scenarios
    
    # Use DummyVecEnv because agent.predict expects a VecEnv-like observation handling
    # and agent.load might also expect a VecEnv if it was trained on one.
    drl_eval_env = DummyVecEnv([lambda: WNTRGymEnv(pipes=PIPES_CONFIG, scenarios=eval_env_list, networks_folder=NETWORKS_FOLDER)])

    agent = GraphPPOAgent(drl_eval_env, PIPES_CONFIG) # Pass the DummyVecEnv
    agent.load(trained_model_path)

    drl_metrics_over_time = []
    total_drl_decision_time = 0
    total_drl_simulation_time = 0

    # Determine the number of timesteps (.inp files) in the target scenario
    scenario_path = os.path.join(NETWORKS_FOLDER, target_scenario_name)
    num_inp_files = len([f for f in os.listdir(scenario_path) if f.endswith('.inp')])
    print(f"Scenario '{target_scenario_name}' has {num_inp_files} timesteps (.inp files).")

    # Reset the environment to the specific target scenario and its first timestep
    # The WNTRGymEnv's reset(scenario_name=...) will handle loading Step_0.inp
    # env_method returns a list of results, one for each sub-environment.
    # For DummyVecEnv with one env, it's a list with one item: (obs_dict, info_dict)
    initial_reset_output = drl_eval_env.env_method("reset", scenario_name=target_scenario_name)
    current_obs_dict = initial_reset_output[0][0] # obs from the first env

    for t_step in range(num_inp_files):
        print(f"  DRL - Processing Timestep {t_step + 1}/{num_inp_files} of {target_scenario_name}...")
        
        # The current_obs_dict is already set for the current t_step from the previous iteration's
        # drl_eval_env.step() or the initial reset.
        # WNTRGymEnv's step method, when an episode (scenario timestep) ends,
        # automatically loads the next .inp file if the scenario is not over.

        # --- DRL Decision Making for all pipes in the current network state ---
        # The WNTRGymEnv is designed such that agent makes one action per pipe.
        # We need to loop through all pipes for the current .inp file.
        
        # Get the actual WNTR network model for the current timestep from the env
        # This assumes WNTRGymEnv has an attribute 'current_network'
        # We need to call this on the underlying environment
        current_wn_model_list = drl_eval_env.env_method("get_current_network_model_copy") # Add this method to WNTRGymEnv
        wn_for_this_timestep = current_wn_model_list[0]

        if wn_for_this_timestep is None:
            print(f"    ERROR: Could not get current network model from WNTRGymEnv for timestep {t_step}. Skipping.")
            # Add placeholder metrics or handle error
            drl_metrics_over_time.append({
                'timestep': t_step, 'cost': np.nan, 'pressure_deficit': np.nan,
                'demand_satisfaction': np.nan, 'decision_time': 0, 'simulation_time': 0
            })
            # Try to advance the environment to the next state if possible, or break
            if t_step < num_inp_files - 1:
                # This step is just to advance the internal state of WNTRGymEnv to the next .inp
                # We are not using the DRL action here for *this specific* problem description.
                # The DRL agent makes decisions on a *copy* of the network for evaluation.
                # The environment itself just loads the next .inp file.
                # However, the agent needs *an* action to pass to step.
                # Let's use a "do nothing" for all pipes to advance the env state.
                
                # To advance WNTRGymEnv, it expects one action per pipe.
                # The observation `current_obs_dict` is for the *first* pipe of the current network.
                # We need to simulate the agent making decisions for all pipes to get to the "end of pipe decisions"
                # state within WNTRGymEnv, which then triggers loading the next .inp.

                # This loop simulates the DRL agent making decisions for each pipe in the current network state
                drl_actions_for_this_inp = []
                temp_obs_for_pipe_decisions = current_obs_dict # Start with obs for the first pipe

                time_decision_start = time.perf_counter()
                for _ in range(len(wn_for_this_timestep.pipe_name_list)):
                    # Batch the observation for the agent
                    batched_obs = {key: np.expand_dims(value, axis=0) for key, value in temp_obs_for_pipe_decisions.items()}
                    action_array, _ = agent.predict(batched_obs, deterministic=True)
                    action_for_pipe = action_array[0] # Get the single action
                    
                    # Step the *vectorized* environment with this single action
                    # This will update temp_obs_for_pipe_decisions for the next pipe
                    # and eventually trigger the end of the current .inp file processing in WNTRGymEnv
                    next_obs_vec, _, done_vec, info_vec = drl_eval_env.step(np.array([action_for_pipe]))
                    temp_obs_for_pipe_decisions = {k: v[0] for k,v in next_obs_vec.items()} # Unbatch for next predict

                    # Store the action if needed for applying to wn_for_this_timestep
                    # The action is an index. 0 = no change, 1+ = diameter option index + 1
                    if action_for_pipe > 0:
                        pipe_name_acted_on = wn_for_this_timestep.pipe_name_list[info_vec[0]['pipe_index']-1 if info_vec[0]['pipe_index'] > 0 else 0] # WNTRGymEnv increments pipe_index *after* action
                        chosen_diameter = PIPE_DIAMETER_OPTIONS[action_for_pipe - 1]
                        drl_actions_for_this_inp.append((pipe_name_acted_on, chosen_diameter))

                    if done_vec[0]: # Scenario timestep ended (all pipes processed)
                        current_obs_dict = temp_obs_for_pipe_decisions # This will be obs for next .inp's first pipe
                        break 
                total_drl_decision_time += (time.perf_counter() - time_decision_start)

                # Apply DRL's chosen actions to a *copy* of the network for this timestep
                original_diameters_this_timestep = {p: wn_for_this_timestep.get_link(p).diameter for p in wn_for_this_timestep.pipe_name_list}
                for pipe_name, new_diameter in drl_actions_for_this_inp:
                    if wn_for_this_timestep.get_link(pipe_name): # Check if pipe exists
                         wn_for_this_timestep.get_link(pipe_name).diameter = new_diameter
                
                time_sim_start = time.perf_counter()
                sim_results, sim_metrics = run_epanet_simulation(wn_for_this_timestep) # Simulate the modified copy
                total_drl_simulation_time += (time.perf_counter() - time_sim_start)

                if sim_results and sim_metrics:
                    # Calculate reward components for logging (using the DRL's actions on this timestep's network)
                    # Max_pd and max_cost for the current network state
                    current_max_pd, _ = calculate_max_pd_for_inp(
                        WNTREnvWrapper(pipes=PIPES_CONFIG, scenarios=[target_scenario_name], current_inp_file_for_eval=os.path.join(scenario_path, f"Step_{t_step}.inp"))
                    )
                    
                    # Max cost for this specific network state
                    max_diameter_val = max(p['diameter'] for p in PIPES_CONFIG.values())
                    max_actions_for_max_cost = [(pipe_id, max_diameter_val) for pipe_id in wn_for_this_timestep.pipe_name_list]
                    initial_energy_cost_for_max_cost = sim_metrics.get('total_pump_cost', 0.0) # Use current energy as proxy
                    
                    current_max_cost = compute_total_cost(
                        list(wn_for_this_timestep.pipes()),
                        max_actions_for_max_cost,
                        LABOUR_COST,
                        initial_energy_cost_for_max_cost, # This is an approximation
                        PIPES_CONFIG,
                        original_diameters_this_timestep
                    )


                    _, cost, pd_ratio_metric, demand_sat_metric, _, _, _ = calculate_reward(
                        current_network=wn_for_this_timestep,
                        original_pipe_diameters=original_diameters_this_timestep,
                        actions=drl_actions_for_this_inp,
                        pipes=PIPES_CONFIG,
                        performance_metrics=sim_metrics,
                        labour_cost=LABOUR_COST,
                        downgraded_pipes=False, # DRL is constrained by action mask
                        max_pd=current_max_pd if current_max_pd != float('inf') else sim_metrics.get('total_pressure_deficit', 0) * 2 + 1e-6, # Fallback for max_pd
                        max_cost=current_max_cost if current_max_cost > 0 else cost * 2 + 1e-6 # Fallback for max_cost
                    )
                    pressure_deficit_abs = sim_metrics.get('total_pressure_deficit', np.nan)
                    demand_satisfaction_abs = sim_metrics.get('demand_satisfaction_ratio', np.nan)
                    
                    drl_metrics_over_time.append({
                        'timestep': t_step, 'cost': cost, 'pressure_deficit': pressure_deficit_abs,
                        'demand_satisfaction': demand_satisfaction_abs * 100, # As percentage
                        'decision_time': (time.perf_counter() - time_decision_start),
                        'simulation_time': (time.perf_counter() - time_sim_start)
                    })
                else:
                    print(f"    DRL Simulation failed for timestep {t_step}.")
                    drl_metrics_over_time.append({
                        'timestep': t_step, 'cost': np.nan, 'pressure_deficit': np.nan,
                        'demand_satisfaction': np.nan, 
                        'decision_time': (time.perf_counter() - time_decision_start), 'simulation_time': (time.perf_counter() - time_sim_start)
                    })
            else: # Last timestep
                break
        
    drl_eval_env.close()
    total_drl_overhead = total_drl_decision_time + total_drl_simulation_time
    print(f"DRL Deployment on {target_scenario_name} complete. Total overhead: {total_drl_overhead:.2f}s")
    return pd.DataFrame(drl_metrics_over_time), total_drl_overhead


# --- Helper for GA: Calculate max_pd for a given .inp file ---
class WNTREnvWrapper: # Minimal wrapper to provide methods needed by calculate_max_pd_for_scenario
    def __init__(self, pipes, scenarios, current_inp_file_for_eval=None):
        self.pipes = pipes
        self.pipe_diameter_options = [p['diameter'] for p in pipes.values()]
        self.scenarios = scenarios
        self.networks_folder = NETWORKS_FOLDER
        self.current_scenario = scenarios[0] if scenarios else None
        self.current_inp_file_for_eval = current_inp_file_for_eval # Path to specific .inp

    def simulate_network(self, network: wntr.network.WaterNetworkModel): # Copied from WNTRGymEnv
        try:
            results = run_epanet_simulation(network)
            if results.node['pressure'].isnull().values.any():
                print(f"WARNING: NaN pressure values during utility simulation.")
            metrics = evaluate_network_performance(network, results)
            return results, metrics
        except Exception as e:
            print(f"ERROR: Utility simulation failed. Error: {e}")
            return None, None

def calculate_max_pd_for_inp(env_wrapper_for_inp):
    """ Calculates max_pd for the specific .inp file provided in env_wrapper_for_inp. """
    if not env_wrapper_for_inp.current_inp_file_for_eval:
        return float('inf'), 0
    try:
        wn_temp = wntr.network.WaterNetworkModel(env_wrapper_for_inp.current_inp_file_for_eval)
        min_diameter_val = min(p['diameter'] for p in env_wrapper_for_inp.pipes.values())
        for p_name in wn_temp.pipe_name_list:
            wn_temp.get_link(p_name).diameter = min_diameter_val
        
        temp_results, _ = env_wrapper_for_inp.simulate_network(wn_temp)
        if temp_results and not temp_results.node['pressure'].isnull().values.any():
            temp_metrics = evaluate_network_performance(wn_temp, temp_results)
            return temp_metrics.get('total_pressure_deficit', float('inf')), 1 # 1 sim call
        return float('inf'), 1
    except Exception as e:
        print(f"Error in calculate_max_pd_for_inp for {env_wrapper_for_inp.current_inp_file_for_eval}: {e}")
        return float('inf'), 1

# --- GA Fitness Function (modified to use global context variables) ---
def ga_fitness_function_for_timestep(ga_instance, solution, solution_idx):
    global ga_simulation_calls_counter_for_fitness # Use the specific counter

    ga_simulation_calls_counter_for_fitness += 1

    wn_trial = wntr.network.WaterNetworkModel(ga_current_inp_path_for_fitness) # Use current .inp for this GA run
    
    actions_for_this_solution = []
    downgraded_in_ga_solution = False
    for i, pipe_name in enumerate(wn_trial.pipe_name_list):
        chosen_diameter_index = int(solution[i])
        new_diameter = PIPE_DIAMETER_OPTIONS[chosen_diameter_index]
        
        original_diameter = ga_original_diameters_for_fitness[pipe_name]
        if abs(new_diameter - original_diameter) > 1e-6:
            actions_for_this_solution.append((pipe_name, new_diameter))
        if new_diameter < original_diameter:
            downgraded_in_ga_solution = True
        wn_trial.get_link(pipe_name).diameter = new_diameter

    sim_results, sim_metrics = run_epanet_simulation(wn_trial)

    if not sim_results or (hasattr(sim_results.node, 'pressure') and sim_results.node['pressure'].isnull().values.any()):
        return -1e9

    # Max cost for this specific network state
    max_diameter_val = max(p['diameter'] for p in PIPES_CONFIG.values())
    max_actions_for_max_cost = [(pipe_id, max_diameter_val) for pipe_id in wn_trial.pipe_name_list]
    initial_energy_cost_for_max_cost = sim_metrics.get('total_pump_cost', 0.0)
    
    current_max_cost = compute_total_cost(
        list(wn_trial.pipes()),
        max_actions_for_max_cost,
        LABOUR_COST,
        initial_energy_cost_for_max_cost,
        PIPES_CONFIG,
        ga_original_diameters_for_fitness
    )
    current_cost_for_reward_calc = compute_total_cost(
         list(wn_trial.pipes()), actions_for_this_solution, LABOUR_COST, sim_metrics.get('total_pump_cost',0.0), PIPES_CONFIG, ga_original_diameters_for_fitness
    )

    reward_val, _, _, _, _, _, _ = calculate_reward(
        current_network=wn_trial,
        original_pipe_diameters=ga_original_diameters_for_fitness,
        actions=actions_for_this_solution,
        pipes=PIPES_CONFIG,
        performance_metrics=sim_metrics,
        labour_cost=LABOUR_COST,
        downgraded_pipes=downgraded_in_ga_solution,
        disconnections=False,
        max_pd=ga_max_pd_for_fitness if ga_max_pd_for_fitness != float('inf') else sim_metrics.get('total_pressure_deficit', 0)*2 +1e-6,
        max_cost=current_max_cost if current_max_cost > 0 else current_cost_for_reward_calc * 2 + 1e-6
    )
    return float(reward_val)

def evaluate_ga_on_scenario(target_scenario_name: str, ga_generations=50, ga_pop_size=20, ga_mutation_percent=10):
    """
    Runs the GA to optimize pipe sizes for each timestep (.inp file) of a target scenario.
    Collects performance metrics and computational time.
    """
    global ga_current_inp_path_for_fitness, ga_original_diameters_for_fitness
    global ga_max_pd_for_fitness, ga_simulation_calls_counter_for_fitness

    print(f"\n--- Evaluating GA on Scenario: {target_scenario_name} ---")
    
    ga_metrics_over_time = []
    total_ga_overhead_time = 0
    total_ga_simulation_calls_across_scenario = 0

    scenario_path = os.path.join(NETWORKS_FOLDER, target_scenario_name)
    inp_files_names = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])

    for t_step, inp_filename in enumerate(inp_files_names):
        print(f"  GA - Optimizing for Timestep {t_step + 1}/{len(inp_files_names)} ({inp_filename})...")
        
        # Set global context for the GA fitness function for THIS SPECIFIC .inp file
        ga_current_inp_path_for_fitness = os.path.join(scenario_path, inp_filename)
        wn_base_for_this_timestep = wntr.network.WaterNetworkModel(ga_current_inp_path_for_fitness)
        ga_original_diameters_for_fitness = {
            p_name: wn_base_for_this_timestep.get_link(p_name).diameter for p_name in wn_base_for_this_timestep.pipe_name_list
        }
        num_pipes_this_timestep = len(wn_base_for_this_timestep.pipe_name_list)

        # Calculate max_pd for this specific .inp file (one-time setup cost for this .inp)
        # Use the wrapper to pass the specific .inp file path
        env_wrapper = WNTREnvWrapper(pipes=PIPES_CONFIG, scenarios=[target_scenario_name], current_inp_file_for_eval=ga_current_inp_path_for_fitness)
        ga_max_pd_for_fitness, sim_calls_for_max_pd = calculate_max_pd_for_inp(env_wrapper)
        total_ga_simulation_calls_across_scenario += sim_calls_for_max_pd
        
        ga_simulation_calls_counter_for_fitness = 0 # Reset counter for this GA run

        time_ga_run_start = time.perf_counter()
        ga_instance = pygad.GA(
            num_generations=ga_generations,
            num_parents_mating=max(2, int(ga_pop_size * 0.2)),
            fitness_func=ga_fitness_function_for_timestep, # Use the correct fitness function
            sol_per_pop=ga_pop_size,
            num_genes=num_pipes_this_timestep,
            gene_type=int,
            gene_space={'low': 0, 'high': len(PIPE_DIAMETER_OPTIONS) - 1},
            mutation_percent_genes=ga_mutation_percent,
            # on_generation=lambda ga_inst: print(f"    GA Gen {ga_inst.generations_completed}, Best Fitness: {ga_inst.best_solution()[1]:.2f}") # Optional
        )
        ga_instance.run()
        ga_run_time_this_timestep = time.perf_counter() - time_ga_run_start
        total_ga_overhead_time += ga_run_time_this_timestep
        total_ga_simulation_calls_across_scenario += ga_simulation_calls_counter_for_fitness # Add calls from this GA run

        best_ga_solution_chromosome, best_ga_fitness, _ = ga_instance.best_solution()
        
        # Evaluate the best GA solution for this timestep to get detailed metrics
        wn_best_ga_this_timestep = wntr.network.WaterNetworkModel(ga_current_inp_path_for_fitness)
        ga_actions_for_best_this_timestep = []
        downgraded = False
        for i, pipe_name in enumerate(wn_best_ga_this_timestep.pipe_name_list):
            chosen_idx = int(best_ga_solution_chromosome[i])
            new_diam = PIPE_DIAMETER_OPTIONS[chosen_idx]
            original_diam = ga_original_diameters_for_fitness[pipe_name]
            if abs(new_diam - original_diam) > 1e-6:
                ga_actions_for_best_this_timestep.append((pipe_name, new_diam))
            if new_diam < original_diam:
                downgraded = True
            wn_best_ga_this_timestep.get_link(pipe_name).diameter = new_diam
        
        total_ga_simulation_calls_across_scenario += 1 # For this final evaluation simulation
        ga_best_results, ga_best_metrics = run_epanet_simulation(wn_best_ga_this_timestep)
        
        cost_val, pd_abs_val, demand_sat_val = np.nan, np.nan, np.nan
        if ga_best_results and ga_best_metrics:
            cost_val = compute_total_cost(
                list(wn_best_ga_this_timestep.pipes()),
                ga_actions_for_best_this_timestep,
                LABOUR_COST,
                ga_best_metrics['total_pump_cost'],
                PIPES_CONFIG,
                ga_original_diameters_for_fitness
            )
            pd_abs_val = ga_best_metrics.get('total_pressure_deficit', np.nan)
            demand_sat_val = ga_best_metrics.get('demand_satisfaction_ratio', np.nan)

        ga_metrics_over_time.append({
            'timestep': t_step, 'cost': cost_val, 'pressure_deficit': pd_abs_val,
            'demand_satisfaction': demand_sat_val * 100 if pd.notna(demand_sat_val) else np.nan, # As percentage
            'ga_run_time': ga_run_time_this_timestep,
            'ga_sim_calls_this_run': ga_simulation_calls_counter_for_fitness + sim_calls_for_max_pd + 1
        })
        print(f"    GA Timestep {t_step+1} - Fitness: {best_ga_fitness:.2f}, Cost: {cost_val:.2f}, PD: {pd_abs_val:.2f}, Demand Sat: {demand_sat_val*100 if pd.notna(demand_sat_val) else 'N/A'}%, Time: {ga_run_time_this_timestep:.2f}s")

    print(f"GA Evaluation on {target_scenario_name} complete. Total overhead: {total_ga_overhead_time:.2f}s, Total sim calls: {total_ga_simulation_calls_across_scenario}")
    return pd.DataFrame(ga_metrics_over_time), total_ga_overhead_time


def plot_comparison_time_series(drl_df, ga_df, target_scenario_name, output_dir="Results/Plots"):
    """
    Plots the time-series comparison of DRL and GA metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_to_plot = ['cost', 'pressure_deficit', 'demand_satisfaction']
    y_labels = ['Total Cost (£)', 'Total Pressure Deficit (m)', 'Demand Satisfaction (%)']

    for metric, ylabel in zip(metrics_to_plot, y_labels):
        plt.figure(figsize=(12, 6))
        plt.plot(drl_df['timestep'], drl_df[metric], label=f'DRL - {metric}', marker='o', linestyle='-')
        plt.plot(ga_df['timestep'], ga_df[metric], label=f'GA - {metric}', marker='x', linestyle='--')
        
        plt.xlabel(f"Timestep in Scenario '{target_scenario_name}'")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Comparison: DRL vs GA for {target_scenario_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{target_scenario_name}_{metric}_comparison_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Saved time-series plot: {plot_filename}")
        plt.close()

def main_comparison_workflow():
    """
    Orchestrates the DRL training, deployment, GA evaluation, and comparison.
    """
    print("=== DRL vs GA Comparison Workflow ===")

    # --- Configuration ---
    # Define scenarios for DRL agent training
    # For a focused comparison, you might train on a broader set or a specific set
    # that includes or is similar to your target evaluation scenario.
    drl_training_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
        'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        # 'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', # Optionally add Hanoi for broader training
        # 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    drl_total_timesteps = 200000  # Adjust as needed, e.g., 500000 for a longer run
    drl_model_name_prefix = "drl_agent_for_comparison"
    
    # Define the single target scenario for deployment and GA comparison
    target_scenario_for_comparison = 'anytown_sprawling_3' # Example

    ga_generations = 30 # Keep GA runtime reasonable for iterative runs
    ga_population_size = 20
    ga_mutation_percentage = 15

    # 1. Train DRL Agent (or load if already trained)
    # trained_drl_model_path = "agents/drl_agent_for_comparison_YYYYMMDD_HHMMSS.zip" # Specify if loading
    
    # For a fresh run:
    timestamp_train = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_drl_model_path = train_drl_agent(
        training_scenarios=drl_training_scenarios,
        total_timesteps=drl_total_timesteps,
        model_save_name=f"{drl_model_name_prefix}_{timestamp_train}"
    )
    
    if not os.path.exists(trained_drl_model_path):
        print(f"ERROR: Trained DRL model not found at {trained_drl_model_path}. Exiting.")
        return

    # 2. Deploy DRL Agent on the target scenario and collect metrics
    drl_results_df, drl_total_overhead_time = deploy_drl_agent_on_scenario(
        trained_model_path=trained_drl_model_path,
        target_scenario_name=target_scenario_for_comparison
    )

    # 3. Evaluate GA on the target scenario (iterating through its timesteps)
    ga_results_df, ga_total_overhead_time = evaluate_ga_on_scenario(
        target_scenario_name=target_scenario_for_comparison,
        ga_generations=ga_generations,
        ga_pop_size=ga_population_size,
        ga_mutation_percent=ga_mutation_percentage
    )

    # 4. Save results and plot comparisons
    output_results_dir = "Results/ComparisonData"
    output_plots_dir = "Results/ComparisonPlots"
    os.makedirs(output_results_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)
    
    timestamp_results = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    drl_results_df.to_csv(os.path.join(output_results_dir, f"drl_metrics_{target_scenario_for_comparison}_{timestamp_results}.csv"), index=False)
    ga_results_df.to_csv(os.path.join(output_results_dir, f"ga_metrics_{target_scenario_for_comparison}_{timestamp_results}.csv"), index=False)
    print(f"\nDRL and GA metrics saved to {output_results_dir}")

    # --- Summary of Computational Overhead ---
    print("\n--- Computational Overhead Summary ---")
    print(f"Target Scenario: {target_scenario_for_comparison}")
    print(f"DRL Agent Deployment Time (decision + simulation for all timesteps): {drl_total_overhead_time:.2f} seconds")
    print(f"GA Total Run Time (optimization for all timesteps): {ga_total_overhead_time:.2f} seconds")
    
    summary_overhead = pd.DataFrame({
        'Method': ['DRL Deployment', 'GA Optimization'],
        'Total Time (s)': [drl_total_overhead_time, ga_total_overhead_time]
    })
    summary_overhead.to_csv(os.path.join(output_results_dir, f"overhead_summary_{target_scenario_for_comparison}_{timestamp_results}.csv"), index=False)


    # --- Plot Time-Series Comparisons ---
    if not drl_results_df.empty and not ga_results_df.empty:
        plot_comparison_time_series(drl_results_df, ga_results_df, target_scenario_for_comparison, output_plots_dir)
    else:
        print("Skipping plotting due to empty results for DRL or GA.")

    print("\n=== Comparison Workflow Complete ===")


# --- Add a method to WNTRGymEnv to get a copy of the current network ---
# This needs to be done by modifying the WNTRGymEnv class in PPO_Environment.py
# For now, I'll add a placeholder here to indicate what's needed.
# In PPO_Environment.py, add to WNTRGymEnv class:
#
# def get_current_network_model_copy(self):
#     if self.current_network:
#         # Create a deep copy to avoid modifying the environment's internal state unintentionally
#         # This requires careful handling if wn.copy() is not robust enough or if complex objects are involved.
#         # A simpler way for evaluation is to re-load from the .inp path if available.
#         current_inp_path = self.network_states.get(self.current_time_step)
#         if current_inp_path and os.path.exists(current_inp_path):
#             try:
#                 return wntr.network.WaterNetworkModel(current_inp_path)
#             except Exception as e:
#                 print(f"Error creating copy of network from {current_inp_path}: {e}")
#                 return None
#         # Fallback: try to copy the internal model, might not be perfectly isolated
#         # print("Warning: Falling back to direct copy of internal network model for evaluation.")
#         # return self.current_network.copy() if self.current_network else None 
#     return None


if __name__ == "__main__":
    # Set a higher recursion limit if needed for PyGAD or complex scenarios, though often not necessary
    # import sys
    # sys.setrecursionlimit(2000) 

    # Ensure the 'Modified_nets' folder exists in the expected location relative to this script
    if not os.path.isdir(NETWORKS_FOLDER):
        print(f"ERROR: The '{NETWORKS_FOLDER}' directory was not found in the current working directory ({os.getcwd()}).")
        print("Please ensure your .inp files are structured correctly.")
    else:
        main_comparison_workflow()