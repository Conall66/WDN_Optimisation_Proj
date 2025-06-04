import numpy as np
import torch
import time
import multiprocessing as mp
import os
import datetime
import matplotlib.pyplot as plt
import tqdm
import wntr

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Plot_Agents import PlottingCallback, plot_training_and_performance, plot_action_analysis, plot_final_agent_rewards_by_scenario, plot_upgrades_per_timestep
from Visualise_network import plot_pipe_diameters_heatmap_over_time
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import reward_minimise_pd, reward_pd_and_cost, reward_full_objective, calculate_reward, compute_total_cost

def train_agent_with_monitoring(net_type = 'both', time_steps = 50000):
    """
    Main training function for the GNN-based PPO agent with monitoring.
    """
    # Configuration
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    if net_type == 'both':
    
        scenarios = [
            'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
            'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
            'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
            'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
        ]

    elif net_type == 'hanoi':

        scenarios = [
            'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
            'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
        ]

    elif net_type == 'anytown':

        scenarios = [
            'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
            'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3'
        ]

    # Helper function to create the environment
    def make_env():
        env = WNTRGymEnv(pipes, scenarios) 
        return env

    num_cpu = mp.cpu_count()
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)]) # This line parallelises code
    vec_env = DummyVecEnv([make_env])
    
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.9, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 1
    }
    
    agent = GraphPPOAgent(vec_env, pipes, **ppo_config)
    
    # Instantiate the callback
    plotting_callback = PlottingCallback() # This will log data during training

    print("Starting training with monitoring...")
    start_time = time.time()
    agent.train(total_timesteps=time_steps, callback=plotting_callback)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script = os.path.dirname(__file__)
    agents_dir = os.path.join(script, "agents")
    model_path = os.path.join(agents_dir, f"trained_gnn_ppo_wn_{timestamp}")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    vec_env.close()
    
    return model_path, pipes, scenarios

def evaluate_agent_by_scenario(model_path, pipes, scenarios, num_episodes_per_scenario=3):
    """
    Evaluates the trained agent for each scenario and collects the average rewards.
    """
    print(f"\nEvaluating DRL agent from {os.path.basename(model_path)}...")

    # def make_eval_env():
    #      env = SimpleWNTRGymEnv(
    #         pipes_config=PIPES_CONFIG,
    #         target_scenario_name=TARGET_SCENARIO_NAME, # Crucially, evaluate on the same scenario
    #         networks_folder=NETWORKS_FOLDER_PATH,
    #         labour_cost_per_meter=LABOUR_COST_PER_METER
    #     )
    #      return env
    
    eval_env = DummyVecEnv([lambda: WNTRGymEnv(pipes, scenarios)])
    # eval_env = SubprocVecEnv([]) # This line parallelises code
    
    agent = GraphPPOAgent(eval_env, pipes)
    agent.load(model_path)

    scenario_rewards = {}

    for scenario in scenarios:
        print(f"  - Evaluating scenario: {scenario}")
        episode_rewards = []
        for episode in range(num_episodes_per_scenario):

            # Get the initial, un-batched observation from the underlying environment
            reset_output = eval_env.env_method("reset", scenario_name=scenario)
            obs = reset_output[0][0]
            
            obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

            total_reward = 0
            done = False
            while not done:
                # Predict using the correctly batched observation
                action, _ = agent.predict(obs, deterministic=True)
                
                # The 'obs' variable is overwritten with the next batched observation from the step method
                obs, reward, done, info = eval_env.step(action)
                
                total_reward += reward[0]
            episode_rewards.append(total_reward)
        
        avg_reward = np.mean(episode_rewards)
        scenario_rewards[scenario] = avg_reward

    eval_env.close()
    return scenario_rewards

def evaluate_random_policy_by_scenario(pipes, scenarios, num_episodes_per_scenario=3):
    """
    Evaluates a random policy for each scenario to provide a baseline.
    """
    print("\nEvaluating Random Policy by scenario...")
    eval_env = WNTRGymEnv(pipes, scenarios)
    scenario_rewards = {}

    for scenario in scenarios:
        print(f"  - Evaluating scenario: {scenario}")
        episode_rewards = []
        for _ in range(num_episodes_per_scenario):

            # reset_output = eval_env.env_method("reset", scenario_name=scenario)
            # # Extract the observation 'obs' from the output of the first (and only) environment.
            # obs = reset_output[0][0]
            obs, _ = eval_env.reset(scenario_name=scenario)

            total_reward = 0
            done = False
            while not done:
                # action_mask = eval_env.get_action_mask()
                # valid_actions = np.where(action_mask)[0]
                action = eval_env.action_space.sample()  # Random action from the action space
                # action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        scenario_rewards[scenario] = avg_reward

    eval_env.close()
    return scenario_rewards

def generate_and_save_plots(model_path, log_path, drl_results, random_results, pipes, scenarios):
    """
    Helper function to generate and save all plots for a given training run.
    """
    print(f"\nGenerating and saving plots for {os.path.basename(model_path)}...")
    
    model_timestamp = os.path.basename(log_path).replace('training_log_', '').replace('.csv', '')

    # Define save directories
    plots_dir = "Plots"
    perf_save_path = os.path.join(plots_dir, "Training_and_Performance")
    action_save_path = os.path.join(plots_dir, "Action_Analysis")
    scenario_save_path = os.path.join(plots_dir, "Reward_by_Scenario")
    
    os.makedirs(perf_save_path, exist_ok=True)
    os.makedirs(action_save_path, exist_ok=True)
    os.makedirs(scenario_save_path, exist_ok=True)

    # Plot 1 & 2: Training and Step-wise Performance
    figs_performance = plot_training_and_performance(log_path)
    if figs_performance and all(fig is not None for fig in figs_performance):
        figs_performance[0].savefig(os.path.join(perf_save_path, f"training_metrics_{model_timestamp}.png"))
        figs_performance[1].savefig(os.path.join(perf_save_path, f"stepwise_performance_{model_timestamp}.png"))
        plt.close(figs_performance[0])
        plt.close(figs_performance[1])
        print("  - Saved training and performance plots.")

    # Plot 3: Action Analysis (this part is unchanged)
    fig_actions = plot_action_analysis(log_path)

    if fig_actions:
        fig_actions.savefig(os.path.join(action_save_path, f"action_analysis_{model_timestamp}.png"))
        plt.close(fig_actions)
        print("  - Saved action analysis plot.")

    # Plot 4: Final Reward by Scenario
    fig_scenarios = plot_final_agent_rewards_by_scenario(drl_results, random_results)
    if fig_scenarios:
        fig_scenarios.savefig(os.path.join(scenario_save_path, f"scenario_rewards_{model_timestamp}.png"))
        plt.close(fig_scenarios)
        print("  - Saved scenario reward comparison plot.")
        
        # Plot 5: Agent Upgrade Frequency per Time Step
    print("  - Generating upgrade frequency plot...")
    fig_actions = plot_upgrades_per_timestep(
        model_path=model_path,
        pipes=pipes,
        scenarios=scenarios,
        num_episodes=24 
    )

    if fig_actions:
        # Give this plot a unique filename to avoid overwriting the other action analysis plot
        upgrades_filename = f"upgrades_per_timestep_{model_timestamp}.png"
        fig_actions.savefig(os.path.join(action_save_path, upgrades_filename))
        plt.close(fig_actions)
        print("  - Saved upgrade frequency plot.")

    if fig_actions:
        fig_actions.savefig(os.path.join(action_save_path, f"action_analysis_{model_timestamp}.png"))
        plt.close(fig_actions)
        print("  - Saved upgrade frequency plot.")

def precalculate_global_normalization_constants(base_inp_path, pipes_config_dict, labour_cost_val):
    # Logic to calculate max_pd (copied & adapted from PPO_Environment.py or GA_Alt_Approach.py)
    wn_pd = wntr.network.WaterNetworkModel(base_inp_path)
    min_diam = min(p['diameter'] for p in pipes_config_dict.values())
    for pname in wn_pd.pipe_name_list: wn_pd.get_link(pname).diameter = min_diam
    # results_pd, metrics_pd = run_epanet_simulation(wn_pd), evaluate_network_performance(wn_pd, run_epanet_simulation(wn_pd)[0]) if run_epanet_simulation(wn_pd)[0] else (None, None)

    results_pd = run_epanet_simulation(wn_pd)
    metrics_pd = evaluate_network_performance(wn_pd, results_pd) if results_pd else None

    global_max_pd = metrics_pd.get('total_pressure_deficit', 1000000.0) if metrics_pd else 1000000.0
    if global_max_pd <= 0: global_max_pd = 1.0

    # Logic to calculate max_cost
    wn_cost = wntr.network.WaterNetworkModel(base_inp_path)
    max_diam_opt = max(p['diameter'] for p in pipes_config_dict.values())
    orig_diams = {pname: wn_cost.get_link(pname).diameter for pname in wn_cost.pipe_name_list}
    max_actions = [(pname, max_diam_opt) for pname in wn_cost.pipe_name_list]

    # res_cost_init, met_cost_init = run_epanet_simulation(wn_cost), evaluate_network_performance(wn_cost, run_epanet_simulation(wn_cost)[0]) if run_epanet_simulation(wn_cost)[0] else (None, None)

    res_cost_init = run_epanet_simulation(wn_cost)
    met_cost_init = evaluate_network_performance(wn_cost, res_cost_init) if res_cost_init else None

    energy_cost_base = met_cost_init.get('total_pump_cost', 0.0) if met_cost_init else 0.0
    global_max_cost = compute_total_cost(list(wn_cost.pipes()), max_actions, labour_cost_val, energy_cost_base, pipes_config_dict, orig_diams)
    if global_max_cost <= 0: global_max_cost = 1000000.0

    print(f"Global Precalculated: max_pd={global_max_pd:.2f}, max_cost={global_max_cost:.2f}")
    return global_max_pd, global_max_cost

def train_multiple():

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.9, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 1
    }

    # Applying a low discount factor so the agent starts to prioritise short term rewwards more greatly

    num_cpu = mp.cpu_count()
    total_timesteps = 2048 # Short run through to chekc functionality
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]

    # ===================================================================
    # --- AGENT 1: Anytown Only ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 1: TRAINING ON ANYTOWN ONLY ###")
    print("="*60)

    vec_env_anytown = SubprocVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios) for _ in range(num_cpu)])
    # vec_env_anytown = DummyVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios)])
    agent1 = GraphPPOAgent(vec_env_anytown, pipes, **ppo_config)
    
    cb1 = PlottingCallback()
    agent1.train(total_timesteps=total_timesteps, callback=cb1)
    
    ts1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path1 = os.path.join("agents", f"agent1_anytown_only_{ts1}")
    log_path1 = os.path.join("Plots", f"training_log_agent1_anytown_only_{ts1}.csv")
    agent1.save(model_path1)
    os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    vec_env_anytown.close()

    print(f"Agent 1 training complete. Model: {model_path1}, Log: {log_path1}")
    drl1_results = evaluate_agent_by_scenario(model_path1, pipes, anytown_scenarios)
    rand1_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(model_path1, log_path1, drl1_results, rand1_results, pipes, anytown_scenarios)

    # ===================================================================
    # --- AGENT 2: Hanoi Only ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 2: TRAINING ON HANOI ONLY ###")
    print("="*60)

    vec_env_hanoi = SubprocVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios) for _ in range(num_cpu)])
    # vec_env_hanoi = DummyVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios)])
    agent2 = GraphPPOAgent(vec_env_hanoi, pipes, **ppo_config)
    
    cb2 = PlottingCallback()
    agent2.train(total_timesteps=total_timesteps, callback=cb2)

    ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path2 = os.path.join("agents", f"agent2_hanoi_only_{ts2}")
    log_path2 = os.path.join("Plots", f"training_log_agent2_hanoi_only_{ts2}.csv")
    agent2.save(model_path2)
    os.rename(os.path.join("Plots", "training_log.csv"), log_path2)
    vec_env_hanoi.close()
    
    print(f"Agent 2 training complete. Model: {model_path2}, Log: {log_path2}")
    drl2_results = evaluate_agent_by_scenario(model_path2, pipes, hanoi_scenarios)
    rand2_results = evaluate_random_policy_by_scenario(pipes, hanoi_scenarios)
    generate_and_save_plots(model_path2, log_path2, drl2_results, rand2_results, pipes, hanoi_scenarios)

    # ===================================================================
    # --- AGENT 3: Anytown then Hanoi (Sequential) ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 3: SEQUENTIAL TRAINING (ANYTOWN -> HANOI) ###")
    print("="*60)
    
    # --- Stage 3a: Train on Anytown ---
    print("\n--- Stage 3a: Training on Anytown ---")
    vec_env_seq_anytown = SubprocVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios) for _ in range(num_cpu)])
    # vec_env_seq_anytown = DummyVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios)])
    agent3_pre = GraphPPOAgent(vec_env_seq_anytown, pipes, **ppo_config)
    
    cb3a = PlottingCallback()
    agent3_pre.train(total_timesteps=total_timesteps, callback=cb3a)

    ts3a = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path3a = os.path.join("agents", f"agent3a_anytown_pre_finetune_{ts3a}")
    log_path3a = os.path.join("Plots", f"training_log_agent3a_anytown_pre_finetune_{ts3a}.csv")
    agent3_pre.save(model_path3a)
    os.rename(os.path.join("Plots", "training_log.csv"), log_path3a)
    vec_env_seq_anytown.close()
    
    print(f"Agent 3 pre-training complete. Model: {model_path3a}, Log: {log_path3a}")
    # Optional: plot intermediate results for stage 3a
    drl3a_results = evaluate_agent_by_scenario(model_path3a, pipes, anytown_scenarios)
    rand3a_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(model_path3a, log_path3a, drl3a_results, rand3a_results, pipes, anytown_scenarios)

    # --- Stage 3b: Fine-tune on Hanoi ---
    print("\n--- Stage 3b: Fine-tuning on Hanoi ---")
    vec_env_seq_hanoi = SubprocVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios) for _ in range(num_cpu)])
    # vec_env_seq_hanoi = DummyVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios)])
    agent3_final = GraphPPOAgent(vec_env_seq_hanoi, pipes, **ppo_config)
    
    print(f"Loading pre-trained model from {model_path3a}...")
    agent3_final.load(model_path3a, env=vec_env_seq_hanoi)
    
    cb3b = PlottingCallback()
    agent3_final.train(total_timesteps=total_timesteps, callback=cb3b)

    ts3b = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path3_final = os.path.join("agents", f"agent3b_final_finetuned_{ts3b}")
    log_path3_final = os.path.join("Plots", f"training_log_agent3b_hanoi_finetuned_{ts3b}.csv")
    agent3_final.save(model_path3_final)
    os.rename(os.path.join("Plots", "training_log.csv"), log_path3_final)
    vec_env_seq_hanoi.close()
    
    print(f"Agent 3 fine-tuning complete. Final Model: {model_path3_final}, Final Log: {log_path3_final}")
    # Evaluate final agent on ALL scenarios to test generalization
    drl3_final_results = evaluate_agent_by_scenario(model_path3_final, pipes, all_scenarios)
    rand3_final_results = evaluate_random_policy_by_scenario(pipes, all_scenarios)
    generate_and_save_plots(model_path3_final, log_path3_final, drl3_final_results, rand3_final_results, pipes, all_scenarios)

    print("\n" + "="*60)
    print("### ALL TRAINING RUNS COMPLETE ###")
    print("="*60)

def train_just_anytown():
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.8, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 2
    }

    # Applying a low discount factor so the agent starts to prioritise short term rewwards more greatly

    num_cpu = mp.cpu_count()
    # num_cpu = 2  # For testing, use only 2 CPU cores

    # print(f"Number of CPU cores available: {num_cpu}")

    total_timesteps = 200000
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]

    # ===================================================================
    # --- AGENT 1: Anytown Only ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 1: TRAINING ON ANYTOWN ONLY ###")
    print("="*60)

    start_time = time.time()

    # vec_env_anytown = SubprocVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios) for _ in range(num_cpu)])
    vec_env_anytown = DummyVecEnv([lambda: WNTRGymEnv(pipes, anytown_scenarios)])
    agent1 = GraphPPOAgent(vec_env_anytown, pipes, **ppo_config)
    
    cb1 = PlottingCallback()
    agent1.train(total_timesteps=total_timesteps, callback=cb1)
    
    ts1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path1 = os.path.join("agents", f"agent1_anytown_only_{ts1}")
    log_path1 = os.path.join("Plots", f"training_log_agent1_anytown_only_{ts1}.csv")
    agent1.save(model_path1)

    os.makedirs("Plots", exist_ok=True)

    # Check if file exists before renaming
    if os.path.exists(os.path.join("Plots", "training_log.csv")):
        os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    else:
        print(f"Warning: Could not find training log file to rename. Will continue without renaming.")

    # os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    # vec_env_anytown.close()

    print(f"Agent 1 training complete. Model: {model_path1}, Log: {log_path1}")
    drl1_results = evaluate_agent_by_scenario(model_path1, pipes, anytown_scenarios)
    rand1_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(model_path1, log_path1, drl1_results, rand1_results, pipes, anytown_scenarios)

    training_time = time.time() - start_time

    plt.show()  # Show plots if running interactively
    print("\n" + "="*60)
    print("### TRAINING COMPLETE ###")
    print("="*60)

def train_just_hanoi():

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.8, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 2
    }

    # num_cpu = mp.cpu_count()
    num_cpu = 4  # For testing, use only 2 CPU cores

    # print(f"Number of CPU cores available: {num_cpu}")

    total_timesteps = 200000
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]

    # From the hanoi networks, pick one to determine input global max pressure and cost value
    base_inp_path = "Modified_nets/hanoi-3.inp"
    labour_cost_val = 100.0  # Example value, adjust as needed
    global_max_pd, global_max_cost = precalculate_global_normalization_constants(base_inp_path, pipes, labour_cost_val)
    print(f"Global Precalculated: max_pd={global_max_pd:.2f}, max_cost={global_max_cost:.2f}")

    # ===================================================================
    # --- AGENT 2: Hanoi Only ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 1: TRAINING ON HANOI ONLY ###")
    print("="*60)

    start_time = time.time()

    vec_env_hanoi = SubprocVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios, current_max_cost=global_max_cost, current_max_pd=global_max_pd) for _ in range(num_cpu)], start_method='spawn')
    # vec_env_hanoi = DummyVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios)])
    agent1 = GraphPPOAgent(vec_env_hanoi, pipes, **ppo_config)
    
    cb1 = PlottingCallback()
    agent1.train(total_timesteps=total_timesteps, callback=cb1)
    
    ts1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path1 = os.path.join("agents", f"agent1_hanoi_only_{ts1}")
    log_path1 = os.path.join("Plots", f"training_log_agent1_hanoi_only_{ts1}.csv")
    agent1.save(model_path1)

    os.makedirs("Plots", exist_ok=True)

    # Check if file exists before renaming
    if os.path.exists(os.path.join("Plots", "training_log.csv")):
        os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    else:
        print(f"Warning: Could not find training log file to rename. Will continue without renaming.")

    # os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    # vec_env_anytown.close()

    print(f"Agent 1 training complete. Model: {model_path1}, Log: {log_path1}")
    drl1_results = evaluate_agent_by_scenario(model_path1, pipes, anytown_scenarios)
    rand1_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(model_path1, log_path1, drl1_results, rand1_results, pipes, anytown_scenarios)

    training_time = time.time() - start_time

    plt.show()  # Show plots if running interactively
    print("\n" + "="*60)
    print("### TRAINING COMPLETE ###")
    print("="*60)

def train_both():

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.9, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 2
    }

    # Applying a low discount factor so the agent starts to prioritise short term rewwards more greatly

    num_cpu = mp.cpu_count()
    # num_cpu = 2  # For testing, use only 2 CPU cores

    # print(f"Number of CPU cores available: {num_cpu}")

    total_timesteps = 500000
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]

    # ===================================================================
    # --- AGENT 3: Both ---
    # ===================================================================
    print("\n" + "="*60)
    print("### AGENT 1: TRAINING ON BOTH NETWORKS ###")
    print("="*60)

    start_time = time.time()

    # vec_env_both = SubprocVecEnv([lambda: WNTRGymEnv(pipes, all_scenarios) for _ in range(num_cpu)])
    vec_env_both = DummyVecEnv([lambda: WNTRGymEnv(pipes, hanoi_scenarios)])
    agent1 = GraphPPOAgent(vec_env_both, pipes, **ppo_config)
    
    cb1 = PlottingCallback()
    agent1.train(total_timesteps=total_timesteps, callback=cb1)
    
    ts1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path1 = os.path.join("agents", f"agent1_hanoi_only_{ts1}")
    log_path1 = os.path.join("Plots", f"training_log_agent1_hanoi_only_{ts1}.csv")
    agent1.save(model_path1)

    os.makedirs("Plots", exist_ok=True)

    # Check if file exists before renaming
    if os.path.exists(os.path.join("Plots", "training_log.csv")):
        os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    else:
        print(f"Warning: Could not find training log file to rename. Will continue without renaming.")

    # os.rename(os.path.join("Plots", "training_log.csv"), log_path1)
    # vec_env_anytown.close()

    print(f"Agent 1 training complete. Model: {model_path1}, Log: {log_path1}")
    drl1_results = evaluate_agent_by_scenario(model_path1, pipes, anytown_scenarios)
    rand1_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(model_path1, log_path1, drl1_results, rand1_results, pipes, anytown_scenarios)

    training_time = time.time() - start_time

    plt.show()  # Show plots if running interactively
    print("\n" + "="*60)
    print("### TRAINING COMPLETE ###")
    print("="*60)

def inspect_agent_actions(model_path: str, pipes: dict, scenarios: list, target_scenario_name: str):
    """
    Loads a trained agent and runs it on a single scenario to inspect its actions step-by-step.

    Args:
        model_path (str): Path to the trained agent model file (e.g., "agents/trained_gnn_ppo_wn_...").
        pipes (dict): The pipes configuration dictionary.
        scenarios (list): The list of all possible scenarios the environment can use.
        target_scenario_name (str): The specific scenario to run for inspection.
    """
    print(f"\n--- Inspecting Agent Actions ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Scenario: {target_scenario_name}\n")

    # 1. Setup Environment and Agent
    # A DummyVecEnv is needed for the agent's constructor, but we'll interact with the base env for inspection.
    eval_env = WNTRGymEnv(pipes, scenarios)
    
    # The agent requires a vectorized environment for initialization
    temp_env = DummyVecEnv([lambda: WNTRGymEnv(pipes, scenarios)])
    agent = GraphPPOAgent(temp_env, pipes_config=pipes) # Pass pipes_config to the agent
    agent.load(model_path)

    # 2. Reset the environment to the target scenario
    obs, info = eval_env.reset(scenario_name=target_scenario_name)
    
    done = False
    major_time_step = 0
    
    while not done:
        # Get information about the current pipe BEFORE the agent takes an action
        current_pipe_index = eval_env.current_pipe_index # The index of the pipe being decided on
        pipe_name = eval_env.pipe_names[current_pipe_index] # The name of the pipe
        original_diameter = eval_env.original_diameters_this_timestep[pipe_name] # The pipe's starting diameter

        # Reshape the single observation to have a batch dimension for the agent
        batched_obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        
        # 3. Get the agent's action
        action, _ = agent.predict(batched_obs, deterministic=True)
        
        # The action is an array, so get the integer value
        action_int = action.item()

        # 4. Translate the action into a human-readable description
        if action_int == 0:
            action_desc = "Do Nothing"
        else:
            # Action `i` corresponds to the `i-1` index in the diameter options list
            new_diameter = eval_env.pipe_diameter_options[action_int - 1]
            if new_diameter > original_diameter:
                action_desc = f"Upgrade to {new_diameter}m"
            else:
                # This case should not happen if the action mask is working correctly
                action_desc = f"Invalid Action: Choose diameter {new_diameter}m (not an upgrade)"


        # 5. Print the agent's decision for the current pipe
        print(f"Major Time Step {major_time_step} | Pipe Decision {current_pipe_index + 1}/{len(eval_env.pipe_names)}:")
        print(f"  - Pipe Under Consideration: '{pipe_name}' (Original Diameter: {original_diameter:.4f}m)")
        print(f"  - Agent's Suggested Action: {action_int} => {action_desc}")

        # Step the environment with the chosen action
        obs, reward, terminated, truncated, info = eval_env.step(action_int)
        done = terminated or truncated

        # If the pipe index has reset to 0, it means we've moved to the next major time step
        if eval_env.current_pipe_index == 0 and not done:
            major_time_step += 1
            print(f"\n--- Network State Advanced (Reward for previous step: {reward:.4f}) ---\n")

    eval_env.close()
    print("\n--- Scenario Inspection Complete ---")
    
if __name__ == "__main__":
    # --- Overall Configuration ---
    
    # train_just_anytown()
    # train_multiple()
    train_just_hanoi()
    # train_both()

    # pipes_config = {
    #     'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
    #     'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    #     'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
    #     'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    #     'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
    #     'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    # }

    # # Anytown scenarios
    # scenarios_list = [
    #     'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
    #     'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3', 
    #     'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
    #     'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    # ]

    # anytown_scenarios = [s for s in scenarios_list if 'anytown' in s]
    # hanoi_scenarios = [s for s in scenarios_list if 'hanoi' in s]

    # saved_model_path = "agents/agent1_hanoi_only_20250603_064211"
    
    # if os.path.exists(saved_model_path + ".zip"):
    # #      inspect_agent_actions(saved_model_path, pipes_config, scenarios_list, target_scenario_name='anytown_sprawling_2')
    #     plot_pipe_diameters_heatmap_over_time(
    #         model_path=saved_model_path,
    #         pipes_config=pipes_config,
    #         scenarios_list=hanoi_scenarios,
    #         num_episodes_for_data=12,  # Number of episodes to average over
    #         target_scenario_name='hanoi_sprawling_2'  # Specify the scenario to visualise

    #     )
    # else:
    #      print(f"Model path not found: {saved_model_path}.zip")