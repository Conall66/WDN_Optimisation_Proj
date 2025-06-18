"""
Main Training, Evaluation, and Plotting Workflow for Water Network Optimization.

This script orchestrates the entire DRL process:
1.  Defines all configurations for the experiments.
2.  Sets up and trains the GNN-based PPO agent using parallel environments.
3.  Saves the trained model and detailed training logs.
4.  Evaluates the final agent's performance against baselines (random policy, no-action).
5.  Generates and saves a suite of plots for comprehensive analysis.
"""
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import os
import datetime
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# --- Import Project Modules ---
from PPO_Environment2 import WNTRGymEnv
from Actor_Critic_Nets3 import GraphPPOAgent
# from Plot_Agent2 import (
#     PlottingCallback, 
#     plot_training_diagnostics, 
#     plot_reward_composition,
#     plot_scenario_performance_comparison,
#     plot_pipe_diameters_heatmap_over_time,
#     calculate_initial_network_rewards,
#     generate_episode_data_for_viz,
#     plot_pipe_upgrade_frequency_over_time,
#     plot_pipe_specific_upgrade_frequency
# )

from Plot_Agent2 import *

# ===================================================================
# 1. CORE CONFIGURATION PARAMETERS
# ===================================================================

# Pipe diameter options and associated costs
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}

# PPO agent hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
    "gamma": 0.95, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 2
}

# Network observation space configuration
NETWORK_CONFIG = {
    'max_nodes': 150, 
    'max_pipes': 200
}

# Reward function configuration
# REWARD_CONFIG = {
#     'mode': 'full_objective',
#     'max_pd_normalization': 5000000.0,  # Expected max pressure deficit (taken from Hanoi before modifications made)
#     'max_cost_normalization': 10000000.0 # Expected max intervention cost (taken from plots of random interventions to get correct order of magnitude)
# }

REWARD_CONFIG = {
    'mode': 'custom_normalized',
    'max_cost_normalization': 1_000_000.0 # COST IS NORMALISED AGAINST SINGLE ACTION
}

# Budget configurations tailored for different network scales
BUDGET_CONFIG_HANOI = {
    "initial_budget_per_step": 500_000.0,
    "start_of_episode_budget": 10_000_000.0,
    "ongoing_debt_penalty_factor": 0.0001,
    "max_debt": 1_000_000.0,
    "labour_cost_per_meter": 100.0
}
    
BUDGET_CONFIG_ANYTOWN = {
    "initial_budget_per_step": 1_000_000.0,
    "start_of_episode_budget": 20_000_000.0,
    "ongoing_debt_penalty_factor": 0.0001,
    "max_debt": 1_000_000.0,
    "labour_cost_per_meter": 100.0
}

# List of all available scenarios
ALL_SCENARIOS = [
    'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
    'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
    'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
    'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
]

# ===================================================================
# 2. HELPER FUNCTIONS (Evaluation and Plotting)
# ===================================================================

def make_env(rank: int, seed: int = 0, **kwargs):
    """Utility function for multiprocessed envs"""
    def _init():
        env = WNTRGymEnv(**kwargs)
        # Note: Seeding is important for reproducibility, but might be complex with WNTR.
        # env.reset(seed=seed + rank) 
        return env
    return _init

def evaluate_policy(model_path: str, eval_env_configs: dict, num_episodes: int = 5, is_random: bool = False) -> dict:
    """Evaluates a policy (DRL or random) for each scenario and returns average rewards."""
    policy_name = "Random Policy" if is_random else "DRL Agent"
    print(f"\nEvaluating {policy_name}...")

    scenario_rewards = {}
    
    for scenario in eval_env_configs['scenarios']:
        print(f"  - Evaluating scenario: {scenario}")
        
        scenario_configs = eval_env_configs.copy()
        scenario_configs['scenarios'] = [scenario]
        
        vec_env = DummyVecEnv([lambda: WNTRGymEnv(**scenario_configs)])
        
        agent = None
        if not is_random:
            # The environment passed during agent creation is temporary; it gets replaced by vec_env when loading.
            agent = GraphPPOAgent(vec_env, pipes_config=PIPES_CONFIG, **PPO_CONFIG)
            agent.load(model_path, env=vec_env) # Pass the vec_env here to link them
        
        episode_rewards = []
        for _ in range(num_episodes):
            obs = vec_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                if is_random:
                    action = [vec_env.action_space.sample()]
                else:
                    # --- KEY CHANGE START ---
                    # 1. Get the action mask from the vectorized environment
                    # env_method returns a list, so we get the first element for our single env.
                    action_masks = vec_env.env_method('action_masks')[0]
                    
                    # 2. Pass the mask to the predict function
                    action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
                    # --- KEY CHANGE END ---
                
                obs, rewards, dones, infos = vec_env.step(action)
                reward = rewards[0]
                done = dones[0]
                
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
        scenario_rewards[scenario] = np.mean(episode_rewards)
        vec_env.close()
    
    return scenario_rewards

# ===================================================================
# 3. MAIN TRAINING WORKFLOW
# ===================================================================

def run_training_experiment(
    training_label: str,
    scenarios: list,
    budget_config: dict,
    total_timesteps: int,
    use_subproc_env: bool = True,
    num_cpu: int = max(1, mp.cpu_count() - 2),
    pre_trained_model: str = None
):
    """
    A complete workflow to train, save, evaluate, and plot results for a DRL agent.
    """
    print("\n" + "="*80)
    print(f"### STARTING EXPERIMENT: {training_label} ###")
    print("="*80)
    
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{training_label}_{timestamp}"
    
    # --- Define directories for logs and models ---
    plots_dir = os.path.join("Plots", model_id)
    agents_dir = "agents"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(agents_dir, exist_ok=True)

    # --- 1. Create the Vectorized Environment ---
    env_configs = {
        'pipes_config': PIPES_CONFIG, 'scenarios': scenarios,
        'network_config': NETWORK_CONFIG, 'budget_config': budget_config,
        'reward_config': REWARD_CONFIG
    }
    
    if use_subproc_env and num_cpu > 1:
        print(f"Using SubprocVecEnv with {num_cpu} processes.")
        vec_env = SubprocVecEnv([make_env(i, **env_configs) for i in range(num_cpu)], start_method='spawn')
    else:
        print("Using DummyVecEnv (single process).")
        vec_env = DummyVecEnv([lambda: WNTRGymEnv(**env_configs)])

    # --- 2. Create or Load the Agent ---
    agent = GraphPPOAgent(vec_env, pipes_config= PIPES_CONFIG, **PPO_CONFIG)
    if pre_trained_model:
        print(f"Loading pre-trained model from: {pre_trained_model}")
        agent.load(pre_trained_model)

    # --- 3. Train the Agent with Logging Callback ---
    print(f"Training for {total_timesteps} timesteps...")
    plotting_callback = PlottingCallback(log_dir=plots_dir)
    agent.train(total_timesteps=total_timesteps, callback=plotting_callback)
    
    training_time = (time.time() - start_time) / 60
    print(f"Training completed in {training_time:.2f} minutes.")

    # --- 4. Save the Final Model ---
    model_path = os.path.join(agents_dir, model_id)
    agent.save(model_path)
    print(f"Final model saved to {model_path}.zip")
    vec_env.close()

    # --- 5. Evaluate and Generate Plots ---
    print("\n--- Post-Training Evaluation and Plot Generation ---")
    log_path = os.path.join(plots_dir, "training_log.csv")
    log_df = pd.read_csv(log_path)

    # Evaluate DRL, Random, and No-Action policies
    drl_results = evaluate_policy(model_path, env_configs, num_episodes=1)
    random_results = evaluate_policy(model_path, env_configs, num_episodes=1, is_random=True)
    initial_rewards = calculate_initial_network_rewards(env_configs)
    
    # Generate and save plots
    print("Generating plots...")

    try:
        fig1 = plot_training_diagnostics(log_df, model_id)
        fig1.savefig(os.path.join(plots_dir, "plot_1_training_diagnostics.png")); plt.close(fig1)
        plt.show()

        fig2 = plot_reward_composition(log_df, model_id)
        fig2.savefig(os.path.join(plots_dir, "plot_2_reward_composition.png")); plt.close(fig2)
        plt.show()

        fig3 = plot_scenario_performance_comparison(drl_results, random_results, initial_rewards, model_id)
        fig3.savefig(os.path.join(plots_dir, "plot_3_scenario_performance.png")); plt.close(fig3)
        plt.show()
        
        # Generate episode data for one representative scenario
        rep_scenario = scenarios[-1]
        print(f"Generating episode data for {rep_scenario}...")
        episode_df = generate_episode_data_for_viz(model_path, env_configs, rep_scenario)

        try:
            print(f"Generating pipe diameter heatmap for {rep_scenario}...")
            fig4 = plot_pipe_diameters_heatmap_over_time(
                model_path=model_path,
                pipes_config=env_configs['pipes_config'],
                scenarios_list=scenarios,
                target_scenario_name=rep_scenario,
                budget_config=budget_config,
                reward_config=env_configs['reward_config'],
                network_config=env_configs['network_config'],
                save_dir=plots_dir
            )
            fig4.savefig(os.path.join(plots_dir, "plot_4_pipe_diameters_heatmap.png")); plt.close(fig4)
            plt.show()
        except Exception as e:
            print(f"Error generating pipe diameter heatmap: {e}")

        fig5 = plot_pipe_upgrade_frequency_over_time(log_df, model_id, window_size=500)
        fig5.savefig(os.path.join(plots_dir, "plot_5_pipe_upgrade_frequency.png")); plt.close(fig5)
        plt.show()
        
        fig6 = plot_pipe_specific_upgrade_frequency(log_df, episode_df, model_id)
        fig6.savefig(os.path.join(plots_dir, "plot_6_specific_pipe_upgrades.png")); plt.close(fig6)
        plt.show()

        fig7 = plot_action_type_frequency(log_df, model_id)
        fig7.savefig(os.path.join(plots_dir, "plot_7_action_type_frequency.png")); plt.close(fig7)
        plt.show()

        fig8 = plot_cumulative_pipe_changes(log_df, model_id)
        fig8.savefig(os.path.join(plots_dir, "plot_8_cumulative_pipe_changes.png")); plt.close(fig8)
        plt.show()

        # Add the new episode stats plot
        fig9 = plot_episode_stats(log_df, model_id)
        fig9.savefig(os.path.join(plots_dir, "plot_9_episode_stats.png")); plt.close(fig9)
        plt.show()

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

    print("\n" + "="*80)
    print(f"### EXPERIMENT {training_label} COMPLETE ###")
    print("="*80)
    return model_path

def test_evaluation_and_plotting(model_path, scenarios, budget_config):
    """
    Test just the evaluation and plotting parts of the training workflow using the existing training log.
    
    Args:
        model_path: Path to the trained model (without .zip extension)
        scenarios: List of scenarios to evaluate on
        budget_config: Budget configuration dictionary
    """
    print("\n" + "="*80)
    print(f"### TESTING EVALUATION AND PLOTTING FOR: {model_path} ###")
    print("="*80)
    
    # Extract model_id from the model path
    model_id = os.path.basename(model_path)
    
    # Define directories
    plots_dir = os.path.join("Plots", model_id)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Environment configs
    env_configs = {
        'pipes_config': PIPES_CONFIG, 'scenarios': scenarios,
        'network_config': NETWORK_CONFIG, 'budget_config': budget_config,
        'reward_config': REWARD_CONFIG
    }
    
    # 5. Evaluate and Generate Plots
    print("\n--- Evaluation and Plot Generation ---")
    
    # Check if training log exists
    log_path = os.path.join(plots_dir, "training_log.csv")
    if not os.path.exists(log_path):
        print(f"Error: Training log not found at {log_path}")
        print("Please make sure your model path is correct and the training log exists.")
        return
    
    # Load the existing training log
    log_df = pd.read_csv(log_path)
    print(f"Successfully loaded training log with {len(log_df)} entries.")

    # Evaluate DRL, Random, and No-Action policies
    print("Evaluating trained agent...")
    drl_results = evaluate_policy(model_path, env_configs, num_episodes=1)
    print("Evaluating random policy...")
    random_results = evaluate_policy(model_path, env_configs, num_episodes=1, is_random=True)
    print("Calculating initial network rewards...")
    initial_rewards = calculate_initial_network_rewards(env_configs)
    
    # Generate and save plots
    try:
        print("Generating plots...")
        
        fig1 = plot_training_diagnostics(log_df, model_id)
        fig1.savefig(os.path.join(plots_dir, "plot_1_training_diagnostics.png")); plt.close(fig1)
        plt.show()

        fig2 = plot_reward_composition(log_df, model_id)
        fig2.savefig(os.path.join(plots_dir, "plot_2_reward_composition.png")); plt.close(fig2)
        plt.show()

        fig3 = plot_scenario_performance_comparison(drl_results, random_results, initial_rewards, model_id)
        fig3.savefig(os.path.join(plots_dir, "plot_3_scenario_performance.png")); plt.close(fig3)
        plt.show()
        
        # Generate episode data for one representative scenario
        rep_scenario = scenarios[-1]
        print(f"Generating episode data for {rep_scenario}...")
        episode_df = generate_episode_data_for_viz(model_path, env_configs, rep_scenario)

        try:
            print(f"Generating pipe diameter heatmap for {rep_scenario}...")
            fig4 = plot_pipe_diameters_heatmap_over_time(
                model_path=model_path,
                pipes_config=env_configs['pipes_config'],
                scenarios_list=scenarios,
                target_scenario_name=rep_scenario,
                budget_config=budget_config,
                reward_config=env_configs['reward_config'],
                network_config=env_configs['network_config'],
                save_dir=plots_dir
            )
            fig4.savefig(os.path.join(plots_dir, "plot_4_pipe_diameters_heatmap.png")); plt.close(fig4)
            plt.show()
        except Exception as e:
            print(f"Error generating pipe diameter heatmap: {e}")

        fig5 = plot_pipe_upgrade_frequency_over_time(log_df, model_id, window_size=500)
        fig5.savefig(os.path.join(plots_dir, "plot_5_pipe_upgrade_frequency.png")); plt.close(fig5)
        plt.show()
        
        fig6 = plot_pipe_specific_upgrade_frequency(log_df, episode_df, model_id)
        fig6.savefig(os.path.join(plots_dir, "plot_6_specific_pipe_upgrades.png")); plt.close(fig6)
        plt.show()

        fig7 = plot_action_type_frequency(log_df, model_id)
        fig7.savefig(os.path.join(plots_dir, "plot_7_action_type_frequency.png")); plt.close(fig7)
        plt.show()

        fig8 = plot_cumulative_pipe_changes(log_df, model_id)
        fig8.savefig(os.path.join(plots_dir, "plot_8_cumulative_pipe_changes.png")); plt.close(fig8)
        plt.show()

        # Add the new episode stats plot
        fig9 = plot_episode_stats(log_df, model_id)
        fig9.savefig(os.path.join(plots_dir, "plot_9_episode_stats.png")); plt.close(fig9)
        plt.show()

        print(f"All plots saved to directory: {plots_dir}")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print(f"### TESTING COMPLETE ###")
    print("="*80)

# ===================================================================
# 4. SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":

    
    # This is crucial for SubprocVecEnv on Windows and macOS
    mp.freeze_support()

    # --- Define Scenario Groups ---
    anytown_scenarios = [s for s in ALL_SCENARIOS if 'anytown' in s]
    hanoi_scenarios = [s for s in ALL_SCENARIOS if 'hanoi' in s]

    # trained_model_path = "agents/Anytown_Only_20250618_063339"  
    
    # # Define which scenarios to evaluate on
    # test_scenarios = anytown_scenarios  # or hanoi_scenarios or ALL_SCENARIOS
    
    # # Test just the evaluation and plotting parts
    # test_evaluation_and_plotting(trained_model_path, test_scenarios, BUDGET_CONFIG_ANYTOWN)

    # --- Run Experiment 1: Train on Hanoi network ---
    # Hanoi is smaller and trains faster, good for a first run.
    # hanoi_model_path = run_training_experiment(
    #     training_label="Hanoi_Only",
    #     scenarios=hanoi_scenarios,
    #     budget_config=BUDGET_CONFIG_HANOI,
    #     total_timesteps=50_000,
    #     use_subproc_env=False
    # )

    # --- Run Experiment 2: Train on Anytown network ---
    # Anytown is larger and more complex.
    anytown_model_path = run_training_experiment(
        training_label="Anytown_Only",
        scenarios=anytown_scenarios,
        budget_config=BUDGET_CONFIG_ANYTOWN,
        total_timesteps=50_000,
        use_subproc_env=False
    )
    
    # --- Optional Experiment 3: Fine-tune the Hanoi model on Anytown scenarios ---
    # This demonstrates transfer learning. Uncomment to run.
    # run_training_experiment(
    #     training_label="Hanoi_Finetuned_on_Anytown",
    #     scenarios=anytown_scenarios,
    #     budget_config=BUDGET_CONFIG_ANYTOWN,
    #     total_timesteps=100_000,
    #     use_subproc_env=True,
    #     pre_trained_model=hanoi_model_path
    # )

    print("\nAll training experiments are complete. Check the 'agents' and 'Plots' folders for results.")
