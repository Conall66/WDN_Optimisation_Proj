import numpy as np
import torch
import time
import multiprocessing as mp
import os

from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Plot_Agents import PlottingCallback, plot_training_and_performance, plot_action_analysis, plot_final_agent_rewards_by_scenario

def train_agent_with_monitoring():
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
    
    scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
        'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
        'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]

    # Helper function to create the environment
    def make_env():
        env = WNTRGymEnv(pipes, scenarios) 
        return env

    num_cpu = mp.cpu_count()
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 1
    }
    
    agent = GraphPPOAgent(vec_env, pipes, **ppo_config)
    
    # Instantiate the callback
    plotting_callback = PlottingCallback()

    print("Starting training with monitoring...")
    start_time = time.time()
    agent.train(total_timesteps=200000, callback=plotting_callback)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    model_path = "trained_gnn_ppo_water_network"
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    vec_env.close()
    
    return model_path, pipes, scenarios

def evaluate_agent_by_scenario(model_path, pipes, scenarios, num_episodes_per_scenario=3):
    """
    Evaluates the trained agent for each scenario and collects the average rewards.
    """
    print("\nEvaluating trained agent by scenario...")
    
    # Load the trained agent
    # Create a single environment for evaluation
    eval_env = WNTRGymEnv(pipes, scenarios)
    agent = GraphPPOAgent(eval_env, pipes)
    agent.load(model_path)

    scenario_rewards = {}

    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")
        episode_rewards = []
        for episode in range(num_episodes_per_scenario):
            # Force the environment to reset to the specific scenario
            obs, _ = eval_env.reset(options={'scenario': scenario})
            total_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)
            print(f"  Episode {episode + 1}/{num_episodes_per_scenario} | Total Reward: {total_reward:.2f}")
        
        avg_reward = np.mean(episode_rewards)
        scenario_rewards[scenario] = avg_reward
        print(f"  Average reward for {scenario}: {avg_reward:.2f}")

    eval_env.close()
    return scenario_rewards


if __name__ == "__main__":
    # 1. Train the agent and log data
    model_path, pipes, scenarios = train_agent_with_monitoring()
    
    # 2. Generate plots from the training log
    print("\nGenerating plots from training log...")
    plot_training_and_performance()
    plot_action_analysis()

    # 3. Evaluate the final agent on each scenario
    scenario_results = evaluate_agent_by_scenario(model_path, pipes, scenarios)

    # 4. Plot the final scenario-based rewards
    print("\nGenerating final agent performance plot...")
    plot_final_agent_rewards_by_scenario(scenario_results)
    
    print("\nAll tasks complete!")