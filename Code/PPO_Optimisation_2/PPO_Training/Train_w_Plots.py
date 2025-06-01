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
    # vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)]) # This line parallelises code
    vec_env = DummyVecEnv([make_env])
    
    ppo_config = {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
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

# Add this new function in Train_w_Plots.py
def evaluate_random_policy_by_scenario(pipes, scenarios, num_episodes_per_scenario=3):
    print("\nEvaluating Random Policy by scenario...")
    eval_env = WNTRGymEnv(pipes, scenarios) # (similar to evaluate_agent_by_scenario)
    scenario_rewards = {}

    for scenario in scenarios: #
        print(f"\nEvaluating Random Policy on scenario: {scenario}")
        episode_rewards = []
        for episode in range(num_episodes_per_scenario): #
            obs, _ = eval_env.reset(options={'scenario': scenario}) #
            total_reward = 0
            done = False
            while not done:
                action_mask = eval_env.get_action_mask() # (method in WNTRGymEnv)
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0 # Fallback if no valid actions (should not happen with 'do nothing')

                obs, reward, terminated, truncated, info = eval_env.step(action) #
                done = terminated or truncated #
                total_reward += reward #
            episode_rewards.append(total_reward) #
            print(f"  Random Policy - Episode {episode + 1}/{num_episodes_per_scenario} | Total Reward: {total_reward:.2f}")

        avg_reward = np.mean(episode_rewards) #
        scenario_rewards[scenario] = avg_reward #
        print(f"  Average Random Policy reward for {scenario}: {avg_reward:.2f}")

    eval_env.close() #
    return scenario_rewards

if __name__ == "__main__":

    # 1. Train the agent and log data

    net_types = ['anytown', 'hanoi']

    for net in net_types:

        model_path, pipes, scenarios = train_agent_with_monitoring(net_type = net, time_steps=500000)
        
        # --- MODIFICATION START: Capture returned figure objects ---
        print("\nGenerating plot data...")

        # Find log file path
        script = os.path.dirname(__file__)
        log_file = os.path.join(script, "Plots", "training_log.csv")

        # Add existence check
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found at {log_file}")
            print("Check that PlottingCallback is correctly saving data during training")
            # Continue with the evaluation even if plots can't be generated
            figs_performance = None
            fig_actions = None
        else:
            figs_performance = plot_training_and_performance(log_file)
            fig_actions = plot_action_analysis(log_file)

        # 3. Evaluate the final agent on each scenario
        drl_scenario_results = evaluate_agent_by_scenario(model_path, pipes, scenarios)
        random_scenario_results = evaluate_random_policy_by_scenario(pipes, scenarios)

        # 4. Plot the final scenario-based rewards
        print("\nGenerating final agent performance plot data...")
        fig_scenarios = plot_final_agent_rewards_by_scenario(drl_scenario_results, random_scenario_results)

        # plt.show()
        
        # --- MODIFICATION: Explicitly save all captured figures ---
        print("\nSaving all generated plots...")
        # Extract timestamp from the model path to name plots consistently
        model_timestamp = os.path.basename(model_path).replace('trained_gnn_ppo_wn_', '')

        # Define save directories
        perf_save_path = os.path.join("Plots", "Training and Performance")
        action_save_path = os.path.join("Plots", "Action Analysis")
        scenario_save_path = os.path.join("Plots", "Reward_by_Scenario")
        
        # Create directories if they don't exist
        os.makedirs(perf_save_path, exist_ok=True)
        os.makedirs(action_save_path, exist_ok=True)
        os.makedirs(scenario_save_path, exist_ok=True)
        
        # Save the figures using the object-oriented method
        if figs_performance:
            figs_performance[0].savefig(os.path.join(perf_save_path, f"training_metrics_{model_timestamp}.png"))
            figs_performance[1].savefig(os.path.join(perf_save_path, f"stepwise_performance_{model_timestamp}.png"))
        if fig_actions:
            fig_actions.savefig(os.path.join(action_save_path, f"action_analysis_{model_timestamp}.png"))
        if fig_scenarios:
            fig_scenarios.savefig(os.path.join(scenario_save_path, f"scenario_rewards_{model_timestamp}.png"))
        
        print("All plots saved.")
        # --- MODIFICATION END ---
        
        print("\nAll tasks complete! Displaying plots...")
        # plt.show()  # This will now display all four figures at once