
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv

# Make sure these modules can be imported from the context of this script
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent

class PlottingCallback(BaseCallback):
    """
    A custom callback that logs additional metrics and saves them to a CSV file
    for later plotting. CORRECTED for modern stable-baselines3 API.
    """
    def __init__(self, verbose=2): # Change to verbose = 1 for real training
        super(PlottingCallback, self).__init__(verbose)
        self.log_data = []

    def _on_step(self) -> bool:
        """
        This method is called after each step in the training loop.
        """
        # Access the 'infos' list from the training loop's local variables.
        # This is the modern way to get info from vectorized environments.
        for info in self.locals.get("infos", []):
            
            # The actual info dict may be nested under 'final_info' at the end of an episode
            custom_info = info.get('final_info', info)
            
            # Only log if the info dictionary is not empty and contains our custom data.
            # This check is important because info is empty on intermediate steps.
            if custom_info and 'reward' in custom_info:
                log_entry = {
                    'timesteps': self.num_timesteps,
                    # Access standard training metrics from the logger
                    'total_reward': self.logger.name_to_value.get('train/reward'),
                    'kl_divergence': self.logger.name_to_value.get('train/approx_kl'),
                    'clip_fraction': self.logger.name_to_value.get('train/clip_fraction'),
                    'entropy_loss': self.logger.name_to_value.get('train/entropy_loss'),
                    
                    # Get custom metrics from the environment's info dictionary
                    'step_reward': custom_info.get('reward'),
                    'cost_of_intervention': custom_info.get('cost_of_intervention'),
                    'pressure_deficit': custom_info.get('pressure_deficit'),
                    'demand_satisfaction': custom_info.get('demand_satisfaction'),
                    'pipe_changes': custom_info.get('pipe_changes'),
                    'downgraded_pipes': custom_info.get('downgraded_pipes'),
                }
                self.log_data.append(log_entry)
        return True

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        It saves the collected data to a CSV file.
        """
        if not self.log_data:
            print("Warning: No data was logged by the PlottingCallback. Check if the environment's info dict is being populated correctly.")
            return
            
        df = pd.DataFrame(self.log_data)
        script = os.path.dirname(__file__)
        plots_dir = os.path.join(script, "Plots")
        os.makedirs(plots_dir, exist_ok=True)
        log_path = os.path.join(plots_dir, "training_log.csv")
        df.to_csv(log_path, index=False)
        print(f"Training log saved to {log_path}")

def plot_training_and_performance(log_file="training_log.csv"): #

    # log_file_path = os.path.join(os.path.dirname(__file__), "Plots", "training_log.csv")
    log_file_path = log_file
    if not os.path.exists(log_file_path): # Use log_file_path
        print(f"Log file not found: {log_file_path}")
        return [None, None] # Return list of Nones

    df_full = pd.read_csv(log_file_path) # Load full data

    if df_full.empty:
        print(f"Log file {log_file_path} is empty. No data to plot.")
        return [None, None]

    # Plot 1: Training Metrics
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10)) #
    fig1.suptitle('Training Progress Metrics', fontsize=16) #

    plot_cols_fig1 = {
        (0, 0): ('total_reward', 'Total Reward vs. Training Duration', 'Total Reward'),
        (0, 1): ('kl_divergence', 'KL Divergence vs. Training Duration', 'KL Divergence'),
        (1, 0): ('clip_fraction', 'Clip Fraction vs. Training Duration', 'Clip Fraction'), # Title corrected
        (1, 1): ('entropy_loss', 'Entropy Loss vs. Training Duration', 'Entropy Loss')
    }

    for (r, c), (col, title, ylabel) in plot_cols_fig1.items():
        df_plot = df_full[['timesteps', col]].dropna()
        if not df_plot.empty:
            axs1[r, c].plot(df_plot['timesteps'], df_plot[col])
        else:
            print(f"No data for '{col}' after dropping NaNs.")
        axs1[r, c].set_title(title) #
        axs1[r, c].set_xlabel('Timesteps') #
        axs1[r, c].set_ylabel(ylabel) #
        axs1[r,c].grid(True)


    plt.tight_layout(rect=[0, 0, 1, 0.96]) #
    # Saving is now handled in Train_w_Plots.py

    # Plot 2: Step-wise Performance Metrics
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10)) #
    fig2.suptitle('Step-wise Performance Metrics', fontsize=16) #

    plot_cols_fig2 = {
        (0,0): ('step_reward', 'Reward by Step', 'Reward', 50), # Added rolling window
        (0,1): ('cost_of_intervention', 'Cost of Intervention by Step', 'Cost', 50), # Added rolling window
        (1,0): ('pressure_deficit', 'Pressure Deficit by Step', 'Pressure Deficit', 50), # Added rolling window
        (1,1): ('demand_satisfaction', 'Demand Satisfaction by Step', 'Demand Satisfaction (%)', 50) # Added rolling window
    }

    for (r, c), (col, title, ylabel, window) in plot_cols_fig2.items():
        df_plot = df_full[['timesteps', col]].dropna()
        if not df_plot.empty:
            axs2[r, c].plot(df_plot['timesteps'], df_plot[col].rolling(window=window, center=True, min_periods=1).mean(), label=f'{ylabel} (Smoothed)')
            axs2[r, c].plot(df_plot['timesteps'], df_plot[col], alpha=0.3, label=f'{ylabel} (Raw)')
            axs2[r, c].legend()
        else:
            print(f"No data for '{col}' after dropping NaNs.")
        axs2[r, c].set_title(title) #
        axs2[r, c].set_xlabel('Timesteps') #
        axs2[r, c].set_ylabel(ylabel) #
        axs2[r,c].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) #
    # Saving is now handled in Train_w_Plots.py

    return [fig1, fig2]

def plot_action_analysis(log_file="training_log.csv"): #
    # ... (script and save_path setup, log_file existence check remains similar) ...
    # log_file_path = os.path.join(os.path.dirname(__file__), "Plots", "training_log.csv") # More robust path
    log_file_path = log_file
    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return None # Return None

    df_full = pd.read_csv(log_file_path) # Load full data

    if df_full.empty:
        print(f"Log file {log_file_path} is empty. No data to plot.")
        return None

    fig, ax1 = plt.subplots(figsize=(12, 8)) # (ax1 for the first y-axis)
    ax1.set_title('Action Analysis Metrics vs. Timesteps', fontsize=16) #
    ax1.set_xlabel('Timesteps') #

    # Plot frequencies on the primary y-axis (ax1)
    ax1.set_ylabel('Frequency / Count', color='tab:blue') # (label changed)

    df_pipe_changes = df_full[['timesteps', 'pipe_changes']].dropna()
    if not df_pipe_changes.empty:
        ax1.plot(df_pipe_changes['timesteps'], df_pipe_changes['pipe_changes'].rolling(window=50, center=True, min_periods=1).mean(), label='Pipe Changes (Smoothed)', color='tab:blue') #

    # Assuming 'upgraded_pipes' is the correct key now from PlottingCallback
    plot_key_upgrades = 'upgraded_pipes' 
    if plot_key_upgrades not in df_full.columns and 'downgraded_pipes' in df_full.columns:
        plot_key_upgrades = 'downgraded_pipes' # Fallback

    if plot_key_upgrades in df_full.columns:
        df_upgraded_pipes = df_full[['timesteps', plot_key_upgrades]].dropna()
        if not df_upgraded_pipes.empty:
            ax1.plot(df_upgraded_pipes['timesteps'], df_upgraded_pipes[plot_key_upgrades].rolling(window=50, center=True, min_periods=1).mean(), label='Upgraded Pipes (Smoothed)', color='tab:cyan', linestyle='--') # (key and label potentially changed)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for step_reward
    ax2 = ax1.twinx()
    ax2.set_ylabel('Step Reward', color='tab:red')

    df_step_reward = df_full[['timesteps', 'step_reward']].dropna()
    if not df_step_reward.empty:
        ax2.plot(df_step_reward['timesteps'], df_step_reward['step_reward'].rolling(window=50, center=True, min_periods=1).mean(), label='Step Reward (Smoothed)', color='tab:red', alpha=0.7) #
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True, axis='x') # Grid for x-axis
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7, color='tab:blue') # Primary y-axis grid
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7, color='tab:red') # Secondary y-axis grid

    fig.tight_layout() #
    # Saving is now handled in Train_w_Plots.py
    return fig

def plot_upgrades_per_timestep(model_path: str, pipes: dict, scenarios: list, num_episodes: int = 10):
    """
    Evaluates a trained agent to plot the frequency of pipe upgrades
    at each environment time step.

    This function runs a specified number of episodes and tracks how many
    upgrades the agent performs at each time step within the episodes.
    It then plots the average and standard deviation of these upgrades.

    Args:
        model_path (str): Path to the trained agent model file (without the .zip extension).
        pipes (dict): The pipes configuration dictionary.
        scenarios (list): The list of scenarios for the environment.
        num_episodes (int): The number of episodes to run for the evaluation.
    """
    print(f"\n--- Analyzing agent upgrade frequency per time step ---")
    print(f"Model: {model_path}.zip")
    print(f"Running for {num_episodes} episodes...")

    # 1. Setup Environment and Agent
    # Use a single, non-vectorized environment for this detailed step-by-step analysis.
    eval_env = WNTRGymEnv(pipes, scenarios)
    
    # The agent must be initialized with a vectorized environment, even if temporary.
    temp_env = DummyVecEnv([lambda: WNTRGymEnv(pipes, scenarios)])
    agent = GraphPPOAgent(temp_env, pipes)
    agent.load(model_path)

    max_timesteps = 50 
    upgrades_data = [[] for _ in range(max_timesteps)]

    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        print(f"  > Episode {episode + 1}/{num_episodes} | Scenario: {eval_env.current_scenario}")
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            # The `info` dictionary contains the 'pipe_changes' key only when a full
            # environment time step (one full pass over the network's pipes) is complete.
            if info.get('pipe_changes') is not None:
                # The time step that just finished is `current_time_step - 1`, as the
                # counter is incremented after the info dictionary is created.
                time_step_index = eval_env.current_time_step - 1
                num_upgrades = info['pipe_changes']
                
                if time_step_index < max_timesteps:
                    upgrades_data[time_step_index].append(num_upgrades)
    
    eval_env.close()

    # 3. Process and Plot the Collected Data
    avg_upgrades = []
    std_upgrades = []
    valid_timesteps = []

    for i in range(max_timesteps):
        if upgrades_data[i]:  # Only process timesteps that were actually reached.
            avg_upgrades.append(np.mean(upgrades_data[i]))
            std_upgrades.append(np.std(upgrades_data[i]))
            valid_timesteps.append(i)

    if not valid_timesteps:
        print("Warning: No pipe upgrade data was collected.")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.errorbar(valid_timesteps, avg_upgrades, yerr=std_upgrades, fmt='-o', color='dodgerblue', capsize=5, label='Avg. Upgrades')
    ax.set_title(f'Agent Upgrade Frequency per Environment Time Step\n(Averaged over {num_episodes} episodes)', fontsize=16)
    ax.set_xlabel('Environment Time Step', fontsize=12)
    ax.set_ylabel('Average Number of Pipe Upgrades', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    # 4. Save the Plot
    plots_dir = os.path.join(os.path.dirname(__file__), "Plots", "Action Analysis")
    os.makedirs(plots_dir, exist_ok=True)
    
    model_name = os.path.basename(model_path)
    save_path = os.path.join(plots_dir, f'upgrades_per_timestep_{model_name}.png')
    # plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    # plt.show()

    return fig  # Return the figure object for further use if needed


# Modify signature and plotting logic
def plot_final_agent_rewards_by_scenario(drl_scenario_rewards: dict, random_scenario_rewards: dict): #
    # ... (script and save_path setup remains similar) ...

    scenarios = list(drl_scenario_rewards.keys()) # (use DRL scenarios as primary)
    drl_rewards = [drl_scenario_rewards.get(s, 0) for s in scenarios] # (modified)
    random_rewards = [random_scenario_rewards.get(s, 0) for s in scenarios] # Get corresponding random rewards

    x = np.arange(len(scenarios))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(18, 9)) # (adjusted size)
    rects1 = ax.bar(x - width/2, drl_rewards, width, label='DRL Agent', color='deepskyblue')
    rects2 = ax.bar(x + width/2, random_rewards, width, label='Random Policy', color='lightcoral')

    ax.set_xlabel('Scenario') #
    ax.set_ylabel('Average Reward') #
    ax.set_title('Final Agent Reward by Scenario vs. Random Policy') # (title updated)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right') #
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout() #
    # Saving is now handled in Train_w_Plots.py
    return fig #