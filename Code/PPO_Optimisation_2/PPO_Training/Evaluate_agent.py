
"""

Whilst the evaluation function is built into the train_w_plots script, this was developed to evaluate agents already designed in the event of a sequencing error

"""

import numpy as np
import torch
import time
import multiprocessing as mp
import os
import datetime
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Plot_Agents import PlottingCallback, plot_training_and_performance, plot_action_analysis, plot_final_agent_rewards_by_scenario


def evaluate_agent_by_scenario(model_path, pipes, scenarios, num_episodes_per_scenario=3):
    """
    Evaluates the trained agent for each scenario and collects the average rewards.
    """
    print(f"\nEvaluating DRL agent from {os.path.basename(model_path)}...")
    
    # Create a single environment for evaluation
    # Use DummyVecEnv to wrap it, which is standard practice for SB3 evaluation
    eval_env = DummyVecEnv([lambda: WNTRGymEnv(pipes, scenarios)])
    
    # The agent needs to be created with the evaluation environment
    agent = GraphPPOAgent(eval_env, pipes)
    # agent.load(model_path, env=eval_env)
    agent.load(model_path)

    scenario_rewards = {}

    for scenario in scenarios:
        print(f"  - Evaluating scenario: {scenario}")
        episode_rewards = []
        for episode in range(num_episodes_per_scenario):
            # The environment automatically cycles through scenarios, but this structure ensures we capture results for each
            # For more targeted scenario evaluation, the WNTRGymEnv reset method would need modification
            # to deterministically select a scenario. For now, we rely on broad evaluation.
            obs = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward[0] # Reward from vectorized env is a list
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
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action_mask = eval_env.get_action_mask()
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        scenario_rewards[scenario] = avg_reward

    eval_env.close()
    return scenario_rewards

def generate_and_save_plots(model_path, log_path, drl_results, random_results):
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

    # Plot 3: Action Analysis
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

def main(agent):

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    num_cpu = mp.cpu_count()
    total_timesteps = 500000
    all_scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3', 'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3', 'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    anytown_scenarios = [s for s in all_scenarios if 'anytown' in s]
    hanoi_scenarios = [s for s in all_scenarios if 'hanoi' in s]

    # ts1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_path1 = os.path.join("agents", f"agent1_anytown_only_{ts1}")
    # log_path1 = os.path.join("Plots", f"training_log_agent1_anytown_only_{ts1}.csv")

    plots_save = os.path.join("Plots", "training_plots")
    model_path1 = os.path.join("agents", f"{agent}.zip")
    log_path1 = os.path.join("Plots", f"training_log_{agent}.csv")

    drl1_results = evaluate_agent_by_scenario(model_path1, pipes, anytown_scenarios)
    rand1_results = evaluate_random_policy_by_scenario(pipes, anytown_scenarios)
    generate_and_save_plots(plots_save, log_path1, drl1_results, rand1_results)

    plt.show()

if __name__ == "__main__":

    agent = 'agent1_anytown_only_20250601_093845'
    main(agent)