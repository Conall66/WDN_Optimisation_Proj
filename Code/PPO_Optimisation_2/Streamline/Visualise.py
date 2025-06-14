
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob

import os
import pandas as pd
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym

# Import the custom environment and GNN policy from your files
from PPO_Env import PPOEnv
from Actor_Critic_CleanRL import GNNPolicy

# --- Helper Functions ---

def setup_environment(network_dir='Networks/Simple_Nets'):
    """Initializes and wraps the custom Gym environment."""
    env = PPOEnv(network_files_dir=network_dir)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def load_agent(model_path, env, device):
    """Loads a trained GNNPolicy agent from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please ensure the MODEL_PATH variable is set correctly."
        )
    agent = GNNPolicy(env.observation_space, env.action_space).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()  # Set the agent to evaluation mode for inference
    print(f"Agent loaded successfully from {model_path}")
    return agent

def run_episode(env, agent=None, device='cpu'):
    """
    Runs a single episode with a given policy (agent or random) and collects data.

    Args:
        env (gym.Env): The environment to run the episode in.
        agent (GNNPolicy, optional): The trained agent. If None, a random policy is used.
        device (str): The device to run torch tensors on.

    Returns:
        dict: A dictionary containing collected data from the episode.
    """
    obs, _ = env.reset()
    terminated = False
    truncated = False

    all_actions = []
    network_rewards = []
    network_indices_for_actions = []
    current_network_index = 0

    while not (terminated or truncated):
        if agent:
            # Prepare observation for the GNNPolicy agent (batch size of 1)
            agent_obs = {
                "graph": [obs['graph']],
                "global_state": torch.tensor(obs['global_state'], dtype=torch.float32).unsqueeze(0).to(device),
                "current_pipe_nodes": torch.tensor(obs['current_pipe_nodes'], dtype=torch.long).unsqueeze(0).to(device)
            }
            with torch.no_grad():
                action_tensor, _, _, _ = agent.get_action_and_value(agent_obs)
                action = action_tensor.cpu().item()
        else:  # Use a random policy
            action = env.action_space.sample()

        all_actions.append(action)
        network_indices_for_actions.append(current_network_index)

        obs, reward, terminated, truncated, info = env.step(action)

        # A reward is only calculated upon the completion of a network
        if info.get('network_completed', False):
            network_rewards.append(reward)
            current_network_index += 1

    return {
        'all_actions': all_actions,
        'network_indices': network_indices_for_actions,
        'network_rewards': network_rewards,
    }

# --- Plotting Functions ---

def plot_training_metrics(wandb_run_path):
    """
    Fetches data from a wandb run and plots key training metrics.

    Args:
        wandb_run_path (str): The path to the wandb run (entity/project/run_id).
    """
    try:
        api = wandb.Api()
        run = api.run(wandb_run_path)
        history = run.history(
            keys=[
                "charts/episodic_return",
                "losses/policy_loss",
                "losses/value_loss",
                "losses/entropy",
                "global_step"
            ]
        )
        print(f"\nSuccessfully fetched data for wandb run: {run.name}")
    except Exception as e:
        print(f"Could not fetch wandb run data from '{wandb_run_path}'. Error: {e}")
        print("Please ensure the WANDB_RUN_PATH is correct and you are logged in to wandb.")
        return

    # Clean up data by dropping rows where the primary metric is NaN
    history = history.dropna(subset=['charts/episodic_return'])

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Performance Metrics', fontsize=18, y=0.95)

    # Episodic Return
    sns.lineplot(data=history, x='global_step', y='charts/episodic_return', ax=axs[0, 0], color='royalblue')
    axs[0, 0].set_title('Episodic Return Over Time')
    axs[0, 0].set_xlabel('Global Timestep')
    axs[0, 0].set_ylabel('Episodic Return')

    # Policy Loss
    sns.lineplot(data=history, x='global_step', y='losses/policy_loss', ax=axs[0, 1], color='crimson')
    axs[0, 1].set_title('Policy Loss Over Time')
    axs[0, 1].set_xlabel('Global Timestep')
    axs[0, 1].set_ylabel('Policy Loss')

    # Value Loss
    sns.lineplot(data=history, x='global_step', y='losses/value_loss', ax=axs[1, 0], color='forestgreen')
    axs[1, 0].set_title('Value Loss Over Time')
    axs[1, 0].set_xlabel('Global Timestep')
    axs[1, 0].set_ylabel('Value Loss')

    # Entropy Loss
    sns.lineplot(data=history, x='global_step', y='losses/entropy', ax=axs[1, 1], color='darkorange')
    axs[1, 1].set_title('Entropy Over Time')
    axs[1, 1].set_xlabel('Global Timestep')
    axs[1, 1].set_ylabel('Entropy')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_action_frequency(agent_data, env):
    """
    Plots the frequency of each action at each network evolution step using a heatmap.

    Args:
        agent_data (dict): Data collected from the trained agent's run.
        env (gym.Env): The environment instance.
    """
    df = pd.DataFrame({
        'network_index': agent_data['network_indices'],
        'action': agent_data['all_actions']
    })
    
    # Action 0 is 'No Change', actions 1-7 correspond to pipe sizes
    action_labels = {0: 'No Change'}
    action_labels.update({i + 1: name for i, name in enumerate(env.pipes.keys())})

    # Create a pivot table to count actions per network
    action_counts = df.pivot_table(index='action', columns='network_index', aggfunc='size', fill_value=0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        action_counts,
        annot=True,
        fmt='d',
        cmap='viridis',
        yticklabels=[action_labels[i] for i in action_counts.index]
    )
    plt.title('Action Frequency per Network Evolution', fontsize=16)
    plt.xlabel('Network Index (Evolution Step)')
    plt.ylabel('Action Taken')
    plt.show()

def plot_reward_comparison(agent_data, random_data):
    """
    Plots the cumulative rewards of the trained agent vs. a random policy.

    Args:
        agent_data (dict): Data collected from the trained agent's run.
        random_data (dict): Data collected from the random policy's run.
    """
    agent_cumulative_rewards = np.cumsum(agent_data['network_rewards'])
    random_cumulative_rewards = np.cumsum(random_data['network_rewards'])
    
    network_indices = np.arange(1, len(agent_cumulative_rewards) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(network_indices, agent_cumulative_rewards, 'o-', label='Trained Agent', color='blue')
    plt.plot(network_indices, random_cumulative_rewards, 's--', label='Random Policy', color='red')
    
    plt.title('Agent vs. Random Policy: Cumulative Reward', fontsize=16)
    plt.xlabel('Network Evolution Step')
    plt.ylabel('Cumulative Reward')
    plt.xticks(network_indices)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def load_tensorboard_data(log_dir, scalars=None):
    """
    Loads tensorboard data from the specified directory.
    
    Args:
        log_dir: Path to the tensorboard log directory
        scalars: List of scalar metrics to extract (None = all available metrics)
        
    Returns:
        Dictionary with metric names as keys and pandas DataFrames as values
    """
    # Find the most recent run directory
    run_dirs = glob.glob(os.path.join(log_dir, "*"))
    if not run_dirs:
        raise ValueError(f"No runs found in {log_dir}")
    
    latest_run = max(run_dirs, key=os.path.getmtime)
    print(f"Loading data from: {latest_run}")
    
    # Load event data
    event_acc = EventAccumulator(latest_run)
    event_acc.Reload()
    
    # Get available tags
    available_scalars = event_acc.Tags()['scalars']
    print(f"Available metrics: {available_scalars}")
    
    # If no specific scalars requested, use all available
    if scalars is None:
        scalars = available_scalars
    
    # Load data for each requested scalar
    data = {}
    for scalar in scalars:
        if scalar in available_scalars:
            events = event_acc.Scalars(scalar)
            data[scalar] = pd.DataFrame(events)[['step', 'value']]
            print(f"Loaded {scalar}: {len(data[scalar])} data points")
        else:
            print(f"Warning: {scalar} not found in tensorboard data")
    
    return data

def plot_training_metrics(log_dir, save_path=None, figsize=(18, 10)):
    """
    Creates plots of key PPO training metrics.
    
    Args:
        log_dir: Path to the tensorboard log directory
        save_path: Path to save the figure (None = show only)
        figsize: Size of the figure as (width, height)
    """
    # Metrics to extract
    metrics = [
        'rollout/ep_rew_mean',           # Episode reward
        'train/clip_fraction',           # PPO clip fraction
        'train/clip_range',              # PPO clip range
        'train/entropy_loss',            # Entropy loss
        'train/explained_variance',      # Explained variance
        'train/learning_rate',           # Learning rate
        'train/policy_gradient_loss',    # Policy gradient loss
        'train/value_loss',              # Value function loss
        'train/approx_kl'                # Approximate KL divergence
    ]
    
    # Load data
    data = load_tensorboard_data(log_dir, metrics)
    
    if not data:
        print("No data found. Check that training has been run and metrics are being logged.")
        return
    
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Plot rewards
    if 'rollout/ep_rew_mean' in data:
        ax = axs[0, 0]
        df = data['rollout/ep_rew_mean']
        ax.plot(df['step'], df['value'])
        ax.set_title('Mean Episode Reward', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # Plot KL divergence
    if 'train/approx_kl' in data:
        ax = axs[0, 1]
        df = data['train/approx_kl']
        ax.plot(df['step'], df['value'])
        ax.set_title('Approximate KL Divergence', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('KL Divergence')
        ax.grid(True, alpha=0.3)
    
    # Plot entropy loss
    if 'train/entropy_loss' in data:
        ax = axs[1, 0]
        df = data['train/entropy_loss']
        ax.plot(df['step'], df['value'])
        ax.set_title('Entropy Loss', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot clip fraction
    if 'train/clip_fraction' in data:
        ax = axs[1, 1]
        df = data['train/clip_fraction']
        ax.plot(df['step'], df['value'])
        ax.set_title('Clip Fraction', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Fraction')
        ax.grid(True, alpha=0.3)
    
    # Plot value loss
    if 'train/value_loss' in data:
        ax = axs[2, 0]
        df = data['train/value_loss']
        ax.plot(df['step'], df['value'])
        ax.set_title('Value Function Loss', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot policy gradient loss
    if 'train/policy_gradient_loss' in data:
        ax = axs[2, 1]
        df = data['train/policy_gradient_loss']
        ax.plot(df['step'], df['value'])
        ax.set_title('Policy Gradient Loss', fontsize=14)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle('PPO Training Metrics', fontsize=16)
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
        plt.show()

# --- Main Execution Block ---

if __name__ == '__main__':

    MODEL_PATH = "ppo_gnn_wntr_model.zip"  # Update this path to your model file
    WANDB_RUN_PATH = "wandb/run-20250613_164047-vmqi0ynt/run-vmqi0ynt.wandb"  # Update this to your wandb run path

    # Verify that placeholder paths have been updated
    if "your_entity" in WANDB_RUN_PATH or "path/to/your" in MODEL_PATH:
        print("="*60)
        print("ERROR: Please update the WANDB_RUN_PATH and MODEL_PATH variables")
        print("in the script before running.")
        print("="*60)
    else:
        # Use CPU for analysis, as it's not computationally intensive
        device = torch.device("cpu")

        # 1. Plot training metrics from wandb
        plot_training_metrics(WANDB_RUN_PATH)

        # 2. Set up environment and load agent for simulation
        env = setup_environment()
        agent = load_agent(MODEL_PATH, env, device)

        # 3. Run episodes to collect data for agent and random policy
        print("\nRunning episode with trained agent...")
        agent_run_data = run_episode(env, agent=agent, device=device)
        print("Running episode with random policy...")
        random_run_data = run_episode(env, agent=None)
        print("Simulations complete.")

        # 4. Generate and display the plots
        print("\nGenerating plots...")
        plot_action_frequency(agent_run_data, env)
        plot_reward_comparison(agent_run_data, random_run_data)
        print("All plots displayed.")