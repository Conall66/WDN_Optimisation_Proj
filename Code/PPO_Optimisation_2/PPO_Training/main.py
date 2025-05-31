"""
Integration script for GNN-based PPO agent with your water distribution network environment

This script shows how to integrate the GNN actor-critic networks with your existing
PPO_Environment.py and train the agent.

Requirements:
pip install torch torch-geometric stable-baselines3 torch-scatter torch-sparse
"""

import numpy as np
import torch
import time
from typing import Dict, List
import matplotlib.pyplot as plt

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets import GraphPPOAgent, GraphAwareWNTREnv

def train_gnn_ppo_agent():
    """
    Main training function for the GNN-based PPO agent
    """
    
    # Configuration from your original script
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
    
    print("Creating water distribution network environment...")
    
    # Create your base environment
    base_env = WNTRGymEnv(pipes, scenarios, max_episodes=1000)
    
    # Wrap with graph-aware functionality
    env = GraphAwareWNTREnv(base_env, pipes)

    env.reset()  # Initialize the environment
    
    print(f"Environment created with action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # PPO hyperparameters optimized for this problem
    ppo_config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,  # Number of steps to run for each environment per update
        "batch_size": 64,  # Minibatch size
        "n_epochs": 10,    # Number of epoch when optimizing the surrogate loss
        "gamma": 0.99,     # Discount factor
        "gae_lambda": 0.95, # Factor for trade-off of bias vs variance for GAE
        "clip_range": 0.2,  # Clipping parameter
        "ent_coef": 0.01,   # Entropy coefficient for the loss calculation
        "vf_coef": 0.5,     # Value function coefficient for the loss calculation
        "max_grad_norm": 0.5, # Maximum value for the gradient clipping
        "verbose": 2
    }
    
    print("Creating GNN-based PPO agent...")
    
    # Create the agent
    agent = GraphPPOAgent(env, pipes, **ppo_config)
    
    print("Starting training...")
    start_time = time.time()
    
    # Train the agent
    total_timesteps = 50000  # Low values for testing (i.e. 50k) - (1e6 for full training)
    agent.train(total_timesteps=total_timesteps)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained agent
    model_path = "trained_gnn_ppo_water_network"
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    return agent, env

def evaluate_trained_agent(agent, env, num_episodes=5):
    """
    Evaluate the trained agent on the environment
    """
    print(f"\nEvaluating trained agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Scenario: {env.env.current_scenario}")
        
        while not done:
            # Get action from trained agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Print progress for current episode
            if info.get('pipe_index', 0) == 0 and info.get('time_step', 0) > 0:
                print(f"  Time step {info['time_step']}: Reward = {reward:.4f}")
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode + 1} completed:")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Episode length: {steps} steps")
    
    # Print summary statistics
    print(f"\nEvaluation Summary:")
    print(f"Average reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths

def plot_training_progress(agent):
    """
    Plot training progress if logging data is available
    """
    # Note: This would require modification to save training metrics during training
    # For now, this is a placeholder for future implementation
    print("Training progress plotting would be implemented here")
    print("Consider using TensorBoard or Weights & Biases for detailed training monitoring")

def compare_with_random_policy(env, num_episodes=3):
    """
    Compare trained agent performance with random policy
    """
    print(f"\nComparing with random policy over {num_episodes} episodes...")
    
    random_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        print(f"Random policy episode {episode + 1}: {env.env.current_scenario}")
        
        while not done:
            # Random action (but respecting action mask)
            action_mask = env.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        random_rewards.append(total_reward)
        print(f"  Random policy total reward: {total_reward:.4f}")
    
    print(f"Random policy average reward: {np.mean(random_rewards):.4f} ± {np.std(random_rewards):.4f}")
    
    return random_rewards

def main():
    """
    Main execution function
    """
    print("=== GNN-PPO Water Distribution Network Optimization ===")
    print("This script trains a Graph Neural Network-based PPO agent")
    print("for optimizing pipe sizing in evolving water distribution networks.\n")
    
    # try:
    # Train the agent
    agent, env = train_gnn_ppo_agent()
    
    # Evaluate the trained agent
    trained_rewards, _ = evaluate_trained_agent(agent, env, num_episodes=3)
    
    # Compare with random policy
    random_rewards = compare_with_random_policy(env, num_episodes=3)
    
    # Print comparison
    print(f"\n=== Performance Comparison ===")
    print(f"Trained GNN-PPO Agent: {np.mean(trained_rewards):.4f} ± {np.std(trained_rewards):.4f}")
    print(f"Random Policy:         {np.mean(random_rewards):.4f} ± {np.std(random_rewards):.4f}")
    
    improvement = (np.mean(trained_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
    print(f"Improvement: {improvement:.1f}%")
    
    # Close environment
    env.close()
    
    print("\n=== Training and Evaluation Complete ===")
    print("Model saved and ready for deployment!")
    
    # except Exception as e:
    #     print(f"Error during training/evaluation: {e}")
    #     print("Please check that all dependencies are installed and paths are correct.")

if __name__ == "__main__":
    main()