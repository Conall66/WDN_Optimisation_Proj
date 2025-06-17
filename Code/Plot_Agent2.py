"""
Comprehensive Plotting, Visualization, and Callback Suite for DRL Water Network Optimization.

This script provides:
1.  A custom Stable Baselines 3 callback (`PlottingCallback`) to log detailed training data.
2.  Functions to generate plots from the logged data for diagnostics and analysis.
3.  Functions to visualize network states (e.g., diameters, pressures, agent decisions).
"""

import os
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from natsort import natsorted
from typing import Dict, List, Any
from copy import deepcopy

from stable_baselines3.common.callbacks import BaseCallback

# Apply a consistent, professional style to all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
PLOT_COLORS = sns.color_palette("colorblind")

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
    "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 1
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
    'max_cost_normalization': 10000000.0 # Still used for cost normalization
}

# Budget configurations tailored for different network scales
BUDGET_CONFIG_HANOI = {
        "initial_budget_per_step": 50000.0,
        "start_of_episode_budget": 100000.0,
        "ongoing_debt_penalty_factor": 0.0001,
        "max_debt": 10000000.0,
        "labour_cost_per_meter": 100.0
    }
    
BUDGET_CONFIG_ANYTOWN = {
        "initial_budget_per_step": 2000000.0,
        "start_of_episode_budget": 4000000.0,
        "ongoing_debt_penalty_factor": 0.0001,
        "max_debt": 20000000.0,
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
# 1. TRAINING CALLBACK
# ===================================================================

class PlottingCallback(BaseCallback):
    """
    A custom callback that logs episode information to a CSV file.
    It captures standard PPO metrics and custom data from the 'info' dict.
    """
    def __init__(self, log_dir: str, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, 'training_log.csv')
        self.log_data = []

    def _on_training_start(self) -> None:
        """Called once at the start of training."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        # Initialize CSV with headers
        pd.DataFrame(columns=[
            'timesteps', 'total_reward', 'kl_divergence', 'entropy_loss', 
            'clip_fraction', 'step_reward', 'pipe_changes', 'cumulative_budget',
            'pressure_deficit', 'demand_satisfaction', 'cost_of_intervention',
            'simulation_success', 'weighted_cost', 'weighted_pd', 'weighted_demand'
        ]).to_csv(self.log_path, index=False)

    def _on_step(self) -> bool:
        """Called at each step in the training loop."""
        # Check for 'dones' signal from VecEnv
        if 'dones' in self.locals and self.locals['dones'].any():
            # Retrieve info from the environment (handle SubprocVecEnv)
            infos = self.locals.get("infos", [])
            
            # Log data from the first environment for simplicity
            info = infos[0] if infos else {}

            # Collect standard PPO metrics from the logger
            log_entry = {
            'timesteps': self.num_timesteps,
            'total_reward': self.logger.name_to_value.get('rollout/ep_rew_mean', 0),
            'kl_divergence': self.logger.name_to_value.get('train/approx_kl', 0),
            'entropy_loss': self.logger.name_to_value.get('train/entropy_loss', 0),
            'clip_fraction': self.logger.name_to_value.get('train/clip_fraction', 0),
            }
            
            # Collect custom metrics from the info dict
            custom_metrics = [
                'step_reward', 'pipe_changes', 'cumulative_budget',
                'pressure_deficit', 'demand_satisfaction', 'cost_of_intervention',
                'simulation_success', 'weighted_cost', 'weighted_pd', 'weighted_demand'
            ]
            for key in custom_metrics:
                log_entry[key] = info.get(key)
            
            self.log_data.append(log_entry)

        # Periodically save to disk to avoid data loss
        if self.num_timesteps % 10240 == 0 and self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_path, mode='a', header=False, index=False)
            self.log_data = [] # Clear memory

        return True

    def _on_training_end(self) -> None:
        """Save any remaining data at the end of training."""
        if self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_path, mode='a', header=False, index=False)
            self.log_data = []

# ===================================================================
# 2. DATA GENERATION HELPERS (for post-training analysis)
# ===================================================================

def calculate_initial_network_rewards(env_configs: dict) -> dict:
    """Calculates the reward for the initial state (no actions) of each scenario."""
    from PPO_Environment2 import WNTRGymEnv
    from Reward2 import calculate_reward
    
    print("Calculating initial network rewards for baseline...")
    initial_rewards = {}
    # Create a single env to iterate through scenarios
    env = WNTRGymEnv(**env_configs)
    
    for scenario in env_configs['scenarios']:
        _, info = env.reset(options={'scenario_name': scenario})
        
        # Calculate reward for the initial state with zero cost
        reward_params = {
            'metrics': info['initial_metrics'],
            'cost_of_intervention': 0,
            **env_configs['reward_config']
        }
        reward, _ = calculate_reward(mode=env.reward_mode, params=reward_params)
        # Approximate total episode reward if no actions were taken
        initial_rewards[scenario] = reward * env.num_time_steps 
    env.close()
    return initial_rewards

def generate_episode_data_for_viz(model_path: str, env_configs: dict, target_scenario: str) -> pd.DataFrame:
    """
    (Corrected Version)
    Runs a trained agent for one episode and returns detailed data for visualization.
    """
    from PPO_Environment2 import WNTRGymEnv
    from Actor_Critic_Nets3 import GraphPPOAgent

    print(f"Generating episode data for visualization: '{target_scenario}'...")

    # Create a specific environment for the target scenario
    single_scenario_configs = env_configs.copy()
    single_scenario_configs['scenarios'] = [target_scenario]
    env = WNTRGymEnv(**single_scenario_configs)

    # Load agent
    agent = GraphPPOAgent(env) # Env is temporary for loading
    agent.load(model_path, env=env)

    obs, info = env.reset(options={'scenario_name': target_scenario})
    done = False
    step_data = []

    # Capture initial state before any actions (Timestep 0)
    initial_step_info = {f'diameter_{p}': env.current_network.get_link(p).diameter for p in env.pipe_names}
    initial_step_info['timestep'] = 0
    initial_step_info['pipe_changes'] = 0
    step_data.append(initial_step_info)

    # Run through the episode step-by-step
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        print(f"Action taken: {action.item()}")  # Log the action for debugging
        obs, _, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        # A major timestep is complete when 'pipe_changes' is in the info dict.
        # This is the point to record the state of the network.
        if 'pipe_changes' in info:
            current_step_info = {f'diameter_{p}': env.current_network.get_link(p).diameter for p in env.pipe_names}
            current_step_info['timestep'] = env.current_time_step  # Use the env's internal timestep
            current_step_info['pipe_changes'] = info.get('pipe_changes', 0)
            step_data.append(current_step_info)

    env.close()
    # Ensure the dataframe is set with the correct index
    df = pd.DataFrame(step_data)
    if not df.empty:
        df = df.set_index('timestep')
    return df

# ===================================================================
# 3. PLOTTING FUNCTIONS
# ===================================================================

def plot_training_diagnostics(log_df: pd.DataFrame, experiment_details: str, roll_window=10):
    """PLOT 1: Core PPO training metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Training Process Diagnostics\n{experiment_details}", fontsize=18, y=0.96)
    
    plot_specs = {
        # 'total_reward': {'ax': axes[0, 0], 'title': 'Mean Episodic Reward', 'color': PLOT_COLORS[0]},
        'step_reward': {'ax': axes[0, 0], 'title': 'Total Step Reward', 'color': PLOT_COLORS[0]},
        'kl_divergence': {'ax': axes[0, 1], 'title': 'KL Divergence', 'color': PLOT_COLORS[1]},
        'entropy_loss': {'ax': axes[1, 0], 'title': 'Entropy Loss', 'color': PLOT_COLORS[2]},
        'clip_fraction': {'ax': axes[1, 1], 'title': 'Clipping Fraction', 'color': PLOT_COLORS[3]}
    }
    
    for col, spec in plot_specs.items():
        if col in log_df.columns and not log_df[col].dropna().empty:
            data = log_df[col].dropna()
            # Use rolling mean for smoothing
            smoothed_data = data.rolling(roll_window, min_periods=1).mean()
            spec['ax'].plot(log_df['timesteps'][data.index], smoothed_data, color=spec['color'], lw=2)
            spec['ax'].set_title(spec['title'], fontsize=14)
            spec['ax'].set_ylabel("Value (Smoothed)")
    
    axes[1, 0].set_xlabel("Training Timesteps", fontsize=12)
    axes[1, 1].set_xlabel("Training Timesteps", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def plot_reward_composition(log_df: pd.DataFrame, experiment_details: str, roll_window=10):
    """PLOT 2: Decomposes the total reward into its weighted components."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Reward Composition Over Time\n{experiment_details}", fontsize=18, y=0.96)
    
    plot_specs = {
        'step_reward': {'ax': axes[0, 0], 'title': 'Total Step Reward', 'color': PLOT_COLORS[0]},
        'weighted_cost': {'ax': axes[0, 1], 'title': 'Weighted Cost Component', 'color': PLOT_COLORS[1]},
        'weighted_pd': {'ax': axes[1, 0], 'title': 'Weighted Pressure Deficit Component', 'color': PLOT_COLORS[2]},
        'weighted_demand': {'ax': axes[1, 1], 'title': 'Weighted Demand Satisfaction Component', 'color': PLOT_COLORS[3]}
    }
    
    for col, spec in plot_specs.items():
        if col in log_df.columns and not log_df[col].dropna().empty:
            data = log_df[col].dropna().rolling(roll_window, min_periods=1).mean()
            spec['ax'].plot(log_df['timesteps'][data.index], data, color=spec['color'], lw=2)
            spec['ax'].set_title(spec['title'], fontsize=14)
            spec['ax'].set_ylabel("Reward Value (Smoothed)")

    axes[1, 0].set_xlabel("Training Timesteps", fontsize=12)
    axes[1, 1].set_xlabel("Training Timesteps", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def plot_scenario_performance_comparison(drl_rewards: dict, random_rewards: dict, initial_rewards: dict, experiment_details: str):
    """PLOT 3: Compares final agent performance against baselines across all scenarios."""
    scenarios = natsorted(list(drl_rewards.keys()))
    results_df = pd.DataFrame({
        'DRL Agent': drl_rewards,
        'Random Policy': random_rewards,
        'No Action (Initial State)': initial_rewards
    }).reindex(scenarios)

    fig, ax = plt.subplots(figsize=(18, 10))
    results_df.plot(kind='bar', ax=ax, color=[PLOT_COLORS[0], PLOT_COLORS[1], 'grey'], width=0.7)
    
    ax.set_title(f"Agent Performance vs. Baselines Across Scenarios\n{experiment_details}", fontsize=18)
    ax.set_ylabel("Average Total Episodic Reward", fontsize=12)
    ax.set_xlabel("Scenario Name", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Policy")
    ax.grid(axis='x')
    plt.tight_layout()
    return fig

def create_network_map_with_indices(wn: wntr.network.WaterNetworkModel, ax=None):
    """
    Creates a 2D visualization of the water network with each pipe labeled by its index.
    
    Args:
        wn: WNTR network model
        ax: Matplotlib axis to plot on (optional)
        
    Returns:
        matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()
    
    # Plot the network
    wntr.graphics.plot_network(wn, ax=ax, node_size=30, link_width=1.5, link_labels=True,)
    
    ax.set_title("Network Map with Pipe Indices", fontsize=14)
    return fig

def plot_pipe_diameters_heatmap_over_time(
    model_path: str,
    pipes_config: dict,
    scenarios_list: list,
    target_scenario_name: str,
    budget_config: dict,
    reward_config: dict,
    network_config: dict,
    num_episodes_for_data: int = 1,
    save_dir: str = "Plots/Pipe_Diameter_Evolution"
):
    """
    Runs a trained agent on a single episode of a specified scenario, 
    records pipe diameters at each major network step, and plots the data as a heatmap.
    Also includes a 2D network map with pipe indices for reference.
    """
    print(f"\n--- Generating Pipe Diameter Evolution Heatmap ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Target Scenario: {target_scenario_name}")

    # Import environment and agent classes here to avoid circular dependencies
    from PPO_Environment2 import WNTRGymEnv
    from Actor_Critic_Nets3 import GraphPPOAgent
    
    # === 1. Setup Environment and Agent ===
    env_configs = {
        'pipes_config': pipes_config,
        'scenarios': scenarios_list,
        'network_config': network_config,
        'budget_config': budget_config,
        'reward_config': reward_config
    }
    
    env = WNTRGymEnv(**env_configs)
    # The agent needs an environment to be instantiated, even if it's temporary
    agent = GraphPPOAgent(env)
    agent.load(model_path, env=env)
    
    # === 2. Run Episode to Collect Data ===
    # Reset the environment specifically to the target scenario
    obs, info = env.reset(options={'scenario_name': target_scenario_name})
    
    # Data structures to hold results
    all_pipe_data = {}
    all_pipe_names = set()
    
    # Capture the initial state (network step 0)
    initial_pipes = {pipe: env.current_network.get_link(pipe).diameter 
                     for pipe in env.current_network.pipe_name_list}
    all_pipe_data[0] = initial_pipes
    all_pipe_names.update(initial_pipes.keys())
    
    done = False
    print("Running agent on scenario to collect pipe diameters at each network step...")
    while not done:
        # Agent takes an action
        action, _ = agent.predict(obs, deterministic=True)
        # Environment steps forward
        obs, _, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # A major network step is complete when 'pipe_changes' is in the info dict.
        if 'pipe_changes' in info:
            current_step = env.current_time_step
            step_pipes = {pipe: env.current_network.get_link(pipe).diameter 
                          for pipe in env.current_network.pipe_name_list}
            all_pipe_data[current_step] = step_pipes
            all_pipe_names.update(step_pipes.keys())
    
    # Save a reference to the final network before closing the environment
    final_network = deepcopy(env.current_network)
    env.close()
    
    # === 3. Structure the Collected Data into a DataFrame ===
    # Sort all unique pipe names numerically for the Y-axis
    sorted_pipe_names = natsorted(list(all_pipe_names))
    max_time_step = max(all_pipe_data.keys())
    
    print(f"Data collection complete. Found {len(sorted_pipe_names)} unique pipes across {max_time_step + 1} network steps.")
    
    # Create an empty DataFrame with the correct dimensions
    pipe_df = pd.DataFrame(index=sorted_pipe_names, columns=range(max_time_step + 1))
    
    # Fill the DataFrame with the collected diameter data
    for step, pipe_data in all_pipe_data.items():
        for pipe_name, diameter in pipe_data.items():
            pipe_df.loc[pipe_name, step] = diameter
            
    # === 4. Create a figure with both heatmap and network map ===
    # Create a figure with two subplots side by side
    fig_height = max(10, len(sorted_pipe_names) / 4)
    fig = plt.figure(figsize=(24, fig_height))
    
    # Create a GridSpec to have different sized subplots
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    fig.patch.set_alpha(0)
    
    # Heatmap on the left (larger)
    ax_heatmap = plt.subplot(gs[0])
    sns.heatmap(
        pipe_df.astype(float),
        ax=ax_heatmap,
        cmap="magma",
        linewidths=0.5,
        annot=False,
        cbar_kws={'label': 'Pipe Diameter (m)'},
        mask=pipe_df.isna()
    )
    
    ax_heatmap.set_title(f"Pipe Diameter Evolution for One Episode\nScenario: {target_scenario_name}", fontsize=18)
    ax_heatmap.set_ylabel("Pipe ID", fontsize=14)
    ax_heatmap.set_xlabel("Network Step", fontsize=14)
    
    # Network map on the right (smaller)
    ax_network = plt.subplot(gs[1])
    create_network_map_with_indices(final_network, ax=ax_network)
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"heatmap_{target_scenario_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Heatmap with network map saved to: {save_path}")
    
    return fig

def plot_network_agent_decisions(wn: wntr.network.WaterNetworkModel, upgraded_pipes: List[str], title: str, ax=None):

    """Simply plots which pipes do not have the same diameter as the original network."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))
    else:
        fig = ax.get_figure()

    ax.set_title(title, fontsize=16)
    
    link_colors = {name: 'lightgray' for name in wn.pipe_name_list}
    link_widths = {name: 1.5 for name in wn.pipe_name_list}
    
    for pipe_name in upgraded_pipes:
        if pipe_name in link_colors:
            link_colors[pipe_name] = 'crimson'
            link_widths[pipe_name] = 4.0
            
    wntr.graphics.plot_network(wn, ax=ax, node_size=30, link_cmap=link_colors, link_width=link_widths)
    
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2, label='Unchanged Pipe'),
        Line2D([0], [0], color='crimson', lw=4, label='Upgraded Pipe')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='large')
    return fig

def plot_action_frequency(log_df: pd.DataFrame, experiment_details: str, roll_window=100):
    """PLOT 5: Shows how often each pipe was upgraded during training."""
    action_counts = log_df['pipe_changes'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    action_counts.plot(kind='bar', ax=ax, color=PLOT_COLORS[0], width=0.7)
    
    ax.set_title(f"Pipe Upgrade Frequency During Training\n{experiment_details}", fontsize=16)
    ax.set_ylabel("Number of Upgrades", fontsize=12)
    ax.set_xlabel("Pipe ID", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_pipe_upgrade_frequency_over_time(log_df: pd.DataFrame, experiment_details: str, window_size=1000):
    """
    Creates a plot showing the frequency of pipe upgrades over time during training.
    
    Args:
        log_df: DataFrame containing training log data
        experiment_details: String describing the experiment for plot title
        window_size: Number of timesteps to aggregate for each data point
        
    Returns:
        matplotlib figure object
    """
    # Check if we have pipe_changes data
    if 'pipe_changes' not in log_df.columns:
        print("Error: 'pipe_changes' column not found in log data")
        return None
    
    # Create timestep bins
    max_timestep = log_df['timesteps'].max()
    bins = np.arange(0, max_timestep + window_size, window_size)
    
    # Group data by time windows
    log_df['time_bin'] = pd.cut(log_df['timesteps'], bins, labels=bins[:-1])
    
    # Count pipe upgrades per time window (where pipe_changes > 0)
    upgrade_counts = log_df[log_df['pipe_changes'] > 0].groupby('time_bin').size()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the data
    ax.bar(upgrade_counts.index, upgrade_counts.values, width=window_size*0.8, 
           color=PLOT_COLORS[0], alpha=0.8)
    
    # Add a trend line
    if len(upgrade_counts) > 1:
        z = np.polyfit(upgrade_counts.index.astype(float), upgrade_counts.values, 1)
        p = np.poly1d(z)
        ax.plot(upgrade_counts.index, p(upgrade_counts.index.astype(float)), 
                'r--', linewidth=2, label=f'Trend: {z[0]:.5f}x + {z[1]:.1f}')
    
    # Add labels and title
    ax.set_xlabel("Training Timesteps", fontsize=14)
    ax.set_ylabel("Number of Pipe Upgrades", fontsize=14)
    ax.set_title(f"Pipe Upgrade Frequency Over Training Time\n{experiment_details}", fontsize=16)
    
    # Add grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if len(upgrade_counts) > 1:
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_pipe_specific_upgrade_frequency(log_df: pd.DataFrame, episode_df: pd.DataFrame, experiment_details: str):
    """
    Creates a plot showing which specific pipes were upgraded most frequently during training.
    
    Args:
        log_df: DataFrame containing training log data
        episode_df: DataFrame from generate_episode_data_for_viz containing pipe info
        experiment_details: String describing the experiment for plot title
        
    Returns:
        matplotlib figure object
    """
    # Check if we have pipe_changes data
    if 'pipe_changes' not in log_df.columns:
        print("Error: 'pipe_changes' column not found in log data")
        return None
    
    # Get all pipe names from the episode data
    pipe_cols = [col for col in episode_df.columns if 'diameter_' in col]
    pipe_names = [col.replace('diameter_', '') for col in pipe_cols]
    
    # Filter for rows where pipes were upgraded
    upgrades_df = log_df[log_df['pipe_changes'] > 0].copy()
    
    # Create a dictionary to store upgrade counts for each pipe
    pipe_upgrade_counts = {pipe: 0 for pipe in pipe_names}
    
    # Count upgrades for each pipe (assuming the pipe_changes column contains the pipe name or index)
    for pipe_info in upgrades_df['pipe_changes']:
        # This approach depends on how pipe_changes is stored
        # Assuming it's a string with pipe names or a number representing pipe index
        if isinstance(pipe_info, str) and pipe_info in pipe_names:
            pipe_upgrade_counts[pipe_info] += 1
        elif isinstance(pipe_info, (int, float)) and int(pipe_info) < len(pipe_names):
            pipe_upgrade_counts[pipe_names[int(pipe_info)]] += 1
    
    # Create a DataFrame for plotting
    pipe_counts_df = pd.DataFrame({
        'Pipe': pipe_names,
        'Upgrade Count': [pipe_upgrade_counts.get(p, 0) for p in pipe_names]
    })
    
    # Sort by upgrade count
    pipe_counts_df = pipe_counts_df.sort_values('Upgrade Count', ascending=False)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the data
    bars = ax.bar(pipe_counts_df['Pipe'], pipe_counts_df['Upgrade Count'], 
             color=PLOT_COLORS[0], alpha=0.8)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel("Pipe ID", fontsize=14)
    ax.set_ylabel("Number of Upgrades During Training", fontsize=14)
    ax.set_title(f"Frequency of Specific Pipe Upgrades\n{experiment_details}", fontsize=16)
    
    # Adjust x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# Test plotting functions with input trainig log data
# In Plot_Agent2.py, at the end of the file

if __name__ == "__main__":
    # --- Example Usage ---
    
    # Define the scenario and agent you want to test
    target_scenario = 'anytown_sprawling_3'
    model_path = 'agents/Anytown_Only_20250617_180304.zip' # Make sure this path is correct
    
    # Define all necessary configs for environment initialization
    env_configs = {
        'pipes_config': PIPES_CONFIG,
        'scenarios': [target_scenario], # List of scenarios for the env
        'network_config': NETWORK_CONFIG,
        'budget_config': BUDGET_CONFIG_ANYTOWN, # Use the correct budget for the network
        'reward_config': REWARD_CONFIG
    }
    
    save_dir = 'Plots/Test_plotting'
    os.makedirs(save_dir, exist_ok=True)
    
    # Call the modified heatmap function
    fig = plot_pipe_diameters_heatmap_over_time(
        model_path=model_path,
        pipes_config=env_configs['pipes_config'],
        scenarios_list=env_configs['scenarios'],
        target_scenario_name=target_scenario,
        budget_config=env_configs['budget_config'],
        reward_config=env_configs['reward_config'],
        network_config=env_configs['network_config'],
        save_dir=save_dir
    )
    
    plt.show() # Display the plot
