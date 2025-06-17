
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Dict, List  # Add this import line
import wntr

# Make sure these modules can be imported from the context of this script
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

from Reward import calculate_reward

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
        # This is the modern way to get info from vectorised environments.
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
                    'budget_before_step': custom_info.get('budget_before_step'),
                    'budget_exceeded': custom_info.get('budget_exceeded', False),

                    'simulation_success': custom_info.get('simulation_success', None),  # Capture any simulation errors
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
        if col == 'total_reward': # Specific handling for 'total_reward'
            df_plot_sb3_reward = df_full[['timesteps', 'total_reward']].dropna()
            if not df_plot_sb3_reward.empty:
                axs1[r, c].plot(df_plot_sb3_reward['timesteps'], df_plot_sb3_reward['total_reward'], label='Mean Episodic Reward (SB3)')
                current_ylabel = 'Mean Episodic Reward (SB3)'
            else:
                print(f"No data for 'total_reward' (SB3's train/reward). Attempting to plot smoothed 'step_reward' instead.")
                if 'step_reward' in df_full.columns:
                    df_step_plot = df_full[['timesteps', 'step_reward']].dropna()
                    if not df_step_plot.empty:
                        # Determine a reasonable window size for smoothing
                        window_size = min(50, len(df_step_plot) // 10) if len(df_step_plot) > 10 else 1
                        if window_size == 0: window_size = 1
                        
                        axs1[r, c].plot(df_step_plot['timesteps'], 
                                        df_step_plot['step_reward'].rolling(window=window_size, center=True, min_periods=1).mean(), 
                                        label='Step Reward (Env - Smoothed)')
                        current_ylabel = 'Step Reward (Env - Smoothed)'
                        axs1[r, c].legend() # Add legend if using step_reward
                    else:
                        print("No data for 'step_reward' either.")
                        current_ylabel = ylabel # Fallback to original label
                else:
                    print("No 'step_reward' column found in log for fallback.")
                    current_ylabel = ylabel # Fallback to original label
            axs1[r, c].set_ylabel(current_ylabel)

        else: # For other metrics like kl_divergence, clip_fraction, entropy_loss
            df_plot = df_full[['timesteps', col]].dropna()
            if not df_plot.empty:
                axs1[r, c].plot(df_plot['timesteps'], df_plot[col])
            else:
                print(f"No data for '{col}' after dropping NaNs.")
            axs1[r, c].set_ylabel(ylabel)
        
        axs1[r, c].set_title(title)
        axs1[r, c].set_xlabel('Timesteps')
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

def plot_action_analysis(log_file="training_log.csv"):
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
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Step Reward', color='tab:red')

    # df_step_reward = df_full[['timesteps', 'step_reward']].dropna()
    # if not df_step_reward.empty:
    #     ax2.plot(df_step_reward['timesteps'], df_step_reward['step_reward'].rolling(window=50, center=True, min_periods=1).mean(), label='Step Reward (Smoothed)', color='tab:red', alpha=0.7) #
    # ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # NEW CODE: Add a third y-axis for budget information
    ax3 = ax1.twinx()
    # Offset the position to the right to avoid overlapping with ax2
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Budget', color='tab:green')
    
    # Plot budget information if available in the log
    if 'budget_before_step' in df_full.columns:
        df_budget = df_full[['timesteps', 'budget_before_step']].dropna()
        if not df_budget.empty:
            ax3.plot(df_budget['timesteps'], df_budget['budget_before_step'].rolling(window=50, center=True, min_periods=1).mean(), 
                    label='Budget Available (Smoothed)', color='tab:green', linestyle='-.')
    
    # Also plot budget exceeded flags if available
    if 'budget_exceeded' in df_full.columns:
        df_budget_exceeded = df_full[['timesteps', 'budget_exceeded']].dropna()
        if not df_budget_exceeded.empty:
            # Convert boolean to numeric (1 for exceeded, 0 for not exceeded)
            exceeded_numeric = df_budget_exceeded['budget_exceeded'].astype(int)
            # Scale these values to be visible on the budget axis
            max_budget = df_full['budget_before_step'].max() if 'budget_before_step' in df_full.columns else 1
            exceeded_scaled = exceeded_numeric * (max_budget * 0.1)  # Scale to 10% of max budget for visibility
            
            # Plot budget exceeded events as red dots
            exceeded_indices = df_budget_exceeded[exceeded_numeric > 0].index
            if not exceeded_indices.empty:
                ax3.scatter(
                    df_budget_exceeded.loc[exceeded_indices, 'timesteps'], 
                    exceeded_scaled[exceeded_indices],
                    color='red', marker='x', s=50, label='Budget Exceeded'
                )
    
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Add legends - we need to combine all three axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
    ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper right') # Combine legends from ax1 and ax3 only

    ax1.grid(True, axis='x') # Grid for x-axis
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7, color='tab:blue') # Primary y-axis grid
    # ax2.grid(False)  # No grid for reward axis to avoid cluttering
    ax3.grid(True, axis='y', linestyle=':', alpha=0.7, color='tab:green') # Budget y-axis grid

    fig.tight_layout() #
    return fig

def plot_upgrades_per_timestep(model_path: str, 
                               pipes: dict, 
                               scenarios: list, # Full list of scenarios for env initialization
                               num_episodes: int = 10, 
                               target_scenario_name: Optional[str] = None):
    """
    Evaluates a trained agent to plot the frequency of pipe upgrades
    at each environment time step, optionally for a specific scenario.

    Args:
        model_path (str): Path to the trained agent model file (without .zip extension).
        pipes (dict): The pipes configuration dictionary.
        scenarios (list): The list of all possible scenarios the environment can use.
        num_episodes (int): The number of episodes to run for averaging.
        target_scenario_name (Optional[str]): If provided, all evaluation episodes
                                             will run on this specific scenario.
                                             Otherwise, scenarios are chosen by the env's reset logic.
    """
    print(f"\n--- Analyzing agent upgrade frequency per time step ---")
    print(f"Model: {model_path}.zip")
    if target_scenario_name:
        print(f"Target Scenario: {target_scenario_name}")
    print(f"Running for {num_episodes} episodes...")

    # 1. Setup Environment and Agent
    eval_env = WNTRGymEnv(pipes, scenarios) # Initialize with all scenarios
    
    temp_env = DummyVecEnv([lambda: WNTRGymEnv(pipes, scenarios)])
    agent = GraphPPOAgent(temp_env, pipes_config=pipes) # Ensure pipes_config is passed
    agent.load(model_path)

    # 2. Collect Data Across Episodes
    # Scenarios have up to 51 time steps (Step_0 to Step_50).
    max_timesteps = 51 # Corrected to include time step 50
    upgrades_data = [[] for _ in range(max_timesteps)]

    for episode in range(num_episodes):
        # <<< MODIFICATION: Use target_scenario_name in reset >>>
        obs, info = eval_env.reset(scenario_name=target_scenario_name)
        
        current_episode_scenario = eval_env.current_scenario # Get scenario used
        print(f"  > Episode {episode + 1}/{num_episodes} | Scenario: {current_episode_scenario}")
        
        done = False
        terminated = False
        truncated = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            if info.get('pipe_changes') is not None:
                time_step_index = eval_env.current_time_step - 1
                num_upgrades = info['pipe_changes']
                
                if 0 <= time_step_index < max_timesteps:
                    upgrades_data[time_step_index].append(num_upgrades)
                else:
                    print(f"Warning: time_step_index {time_step_index} out of bounds for upgrades_data (max_timesteps: {max_timesteps})")

    eval_env.close()

    # 3. Process and Plot the Collected Data
    avg_upgrades = []
    std_upgrades = []
    valid_timesteps_indices = []

    for i in range(max_timesteps):
        if upgrades_data[i]:
            avg_upgrades.append(np.mean(upgrades_data[i]))
            std_upgrades.append(np.std(upgrades_data[i]))
            valid_timesteps_indices.append(i)

    if not valid_timesteps_indices:
        print("Warning: No pipe upgrade data was collected for the specified parameters.")
        return None # Return None if no data

    # <<< MODIFICATION: Dynamic plot title >>>
    plot_title = f'Agent Upgrade Frequency per Environment Time Step'
    if target_scenario_name:
        plot_title += f'\nScenario: {target_scenario_name}'
    elif len(scenarios) == 1: # If only one scenario was possible for the environment
        plot_title += f'\nScenario: {scenarios[0]}'
    plot_title += f' (Averaged over {num_episodes} episodes)'

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.errorbar(valid_timesteps_indices, avg_upgrades, yerr=std_upgrades, fmt='-o', color='dodgerblue', capsize=5, label='Avg. Upgrades')
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Environment Time Step (0-50)', fontsize=12)
    ax.set_ylabel('Average Number of Pipe Upgrades', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_xticks(np.arange(0, max_timesteps, 5)) # Set x-ticks for better readability
    plt.tight_layout()

    # 4. Save the Plot
    # Ensure plots_dir is correctly formed relative to Plot_Agents.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "Plots", "Action_Analysis")
    os.makedirs(plots_dir, exist_ok=True)
    
    model_name_part = os.path.basename(model_path)
    scenario_name_part = f"_{target_scenario_name}" if target_scenario_name else "_all_scenarios"
    
    save_filename = f'upgrades_per_env_timestep_{model_name_part}{scenario_name_part}.png'
    save_path = os.path.join(plots_dir, save_filename)
    
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    # plt.show() # Keep commented out if called from a larger script like Train_w_Plots.py

    return fig

# Modify signature and plotting logic
def plot_final_agent_rewards_by_scenario(drl_scenario_rewards: dict, random_scenario_rewards: dict): #

    scenarios = list(drl_scenario_rewards.keys()) # (use DRL scenarios as primary)
    initial_rewards = {}  # Dictionary to store initial rewards for each scenario

    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    # Calculate initial rewards for each scenario
    for scenario in scenarios:
        if 'hanoi' in scenario:
            current_network_path = os.path.join(os.path.dirname(__file__), "Modified_nets", "hanoi-3.inp")
        elif 'anytown' in scenario:
            current_network_path = os.path.join(os.path.dirname(__file__), "Modified_nets", "anytown-3.inp")
        else:
            continue  # Skip if not a recognized scenario

        wn = wntr.network.WaterNetworkModel(current_network_path)

        # original_pipe_diameters = {pipe.id: pipe.diameter for pipe in wn.pipes()}
        # # Actions are all 0 for length of pipes 
        # actions = [(pipe.id, 0) for pipe in wn.pipes()]

        original_pipe_ids = []
        original_diameters = []
        for pipe, pipe_data in wn.pipes():
            original_pipe_ids.append(pipe)
            original_diameters.append(pipe_data.diameter)

        original_pipe_diameters = dict(zip(original_pipe_ids, original_diameters))
        actions = [(pipe, 0) for pipe, pipe_data in wn.pipes()]  # Actions are all 0 for length of pipes

        # print(f"Original pipe diameters for scenario '{scenario}': {original_pipe_diameters}")
        # print(f"Actions for scenario '{scenario}': {actions}")

        results = run_epanet_simulation(wn)
        metrics = evaluate_network_performance(wn, results)

        reward_tuple = calculate_reward(wn, 
        original_pipe_diameters,  # Dictionary of original pipe diameters
        actions,                  # List of pipe ID diameter pairs representing the actions
        pipes,                    # Dictionary of pipe types with unit costs
        metrics,
        100,
        False,
        disconnections=False,
        actions_causing_disconnections=None,
        max_pd = 5000000.0,
        max_cost = 2000000.0)

        initial_rewards[scenario] = reward_tuple[0]

    drl_rewards = [(drl_scenario_rewards.get(s, 0)/50) for s in scenarios] # (modified)
    random_rewards = [(random_scenario_rewards.get(s, 0)/50) for s in scenarios] # Get corresponding random rewards
    initial_network_rewards = [initial_rewards.get(s, 0) for s in scenarios]  # Get initial network rewards

    x = np.arange(len(scenarios))  # the label locations
    width = 0.25  # Reduced width to fit three bars side by side

    # assign magma collour scheme to the bars

    fig, ax = plt.subplots(figsize=(18, 9)) # (adjusted size)
    rects1 = ax.bar(x - width, drl_rewards, width, label='DRL Agent') # (color changed)
    rects2 = ax.bar(x, random_rewards, width, label='Random Policy') # (added alpha for transparency)
    rects3 = ax.bar(x + width, initial_network_rewards, width, label='Initial Network') # (added alpha for transparency)

    ax.set_xlabel('Scenario') #
    ax.set_ylabel('Average Reward') #
    ax.set_title('Final Agent Reward by Scenario vs. Random Policy and Initial Network') # (updated title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right') #
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    ax.bar_label(rects3, padding=3, fmt='%.2f')

    fig.tight_layout() #
    # Saving is now handled in Train_w_Plots.py
    return fig #

def test_plot_final_agent_rewards_by_scenario():
    """
    Test function for plot_final_agent_rewards_by_scenario
    Creates sample data and calls the plotting function
    """
    # Create sample scenario rewards for DRL agent
    drl_scenario_rewards = {
        'hanoi_scenario1': 250.0,
        'hanoi_scenario2': 300.0,
        'anytown_scenario1': 200.0,
        'anytown_scenario2': 180.0
    }
    
    # Create sample scenario rewards for random policy
    random_scenario_rewards = {
        'hanoi_scenario1': 100.0,
        'hanoi_scenario2': 120.0,
        'anytown_scenario1': 80.0,
        'anytown_scenario2': 90.0
    }
    
    # Call the plotting function
    fig = plot_final_agent_rewards_by_scenario(drl_scenario_rewards, random_scenario_rewards)
    
    # Save the plot
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "Plots", "Scenario_Comparison")
    os.makedirs(plots_dir, exist_ok=True)
    
    save_path = os.path.join(plots_dir, "scenario_rewards_comparison_test.png")
    fig.savefig(save_path)
    print(f"Test plot saved to {save_path}")
    
    # Display the plot
    plt.show()
    
    return fig

def plot_simulation_errors_against_reward(training_log, save_path=None):
    """
    Plots the simulation success (boolean) and reward by time step from the training log
    """
    
    log_file_path = training_log
    if not os.path.exists(log_file_path): # Use log_file_path
        print(f"Log file not found: {log_file_path}")
        return [None, None] # Return list of Nones

    df_full = pd.read_csv(log_file_path) # Load full data

    if df_full.empty:
        print(f"Log file {log_file_path} is empty. No data to plot.")
        return [None, None]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title('Simulation Success and Reward by Time Step', fontsize=16)
    ax1.set_xlabel('Timesteps')

    # Plot simulation success on the primary y-axis (ax1)
    ax1.set_ylabel('Simulation Success', color='tab:blue')

    df_success = df_full[['timesteps', 'simulation_success']].dropna()

    if not df_success.empty:
        # Convert boolean to 0/1 if needed
        if df_success['simulation_success'].dtype == bool:
            df_success['simulation_success'] = df_success['simulation_success'].astype(int)
        
        # Create a scatter plot for simulation success (0 = False, 1 = True)
        ax1.scatter(df_success['timesteps'], df_success['simulation_success'], 
                   label='Simulation Success', color='tab:blue', alpha=0.5, s=10)
        
        # Add a rolling average line to show the success rate trend
        window_size = min(100, len(df_success) // 10) if len(df_success) > 10 else 1
        if window_size == 0: window_size = 1
        
        rolling_success = df_success['simulation_success'].rolling(window=window_size, 
                                                                  center=True, 
                                                                  min_periods=1).mean()
        ax1.plot(df_success['timesteps'], rolling_success, 
                label=f'Success Rate (Avg over {window_size} steps)', 
                color='darkblue', linewidth=2)
        
        # Set y-axis limits for simulation success (with some padding)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Failure', 'Success'])
    else:
        print("No data for 'simulation_success' after dropping NaNs.")

    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for reward
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reward', color='tab:red')

    df_reward = df_full[['timesteps', 'step_reward']].dropna()

    if not df_reward.empty:
        ax2.plot(df_reward['timesteps'], df_reward['step_reward'], 
                label='Step Reward', color='tab:red', alpha=0.7)
        
        # Add a rolling average line for rewards
        window_size = min(100, len(df_reward) // 10) if len(df_reward) > 10 else 1
        if window_size == 0: window_size = 1
        
        rolling_reward = df_reward['step_reward'].rolling(window=window_size, 
                                                         center=True, 
                                                         min_periods=1).mean()
        ax2.plot(df_reward['timesteps'], rolling_reward, 
                label=f'Reward (Avg over {window_size} steps)', 
                color='darkred', linewidth=2)
    else:
        print("No data for 'step_reward' after dropping NaNs.")
    
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

    ax1.grid(True, axis='x')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7, color='tab:blue')
    ax2.grid(False)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        print("No save path provided. Plot will not be saved.")
    
    return fig

if __name__ == "__main__":
    # Example usage of the plotting functions
    # This block is for testing purposes and can be removed in production code.

    # Plot action analysis

    # Test scenario reward plot
    test_plot_final_agent_rewards_by_scenario()