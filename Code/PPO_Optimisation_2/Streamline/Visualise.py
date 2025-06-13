
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob

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

if __name__ == "__main__":
    # Path to tensorboard logs
    log_dir = "./ppo_gnn_wntr_tensorboard/"
    
    # Plot and save the metrics
    plot_training_metrics(log_dir, save_path="ppo_training_metrics.png")
    
    # You can also save individual metrics plots if needed
    # For example, to just plot the rewards:
    data = load_tensorboard_data(log_dir, ['rollout/ep_rew_mean'])
    if 'rollout/ep_rew_mean' in data:
        plt.figure(figsize=(10, 6))
        df = data['rollout/ep_rew_mean']
        plt.plot(df['step'], df['value'])
        plt.title('Mean Episode Reward Over Time', fontsize=14)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig("reward_plot.png", dpi=300, bbox_inches='tight')
        plt.show()