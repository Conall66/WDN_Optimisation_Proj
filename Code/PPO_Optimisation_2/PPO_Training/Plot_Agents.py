
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime

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

def plot_training_and_performance(log_file="training_log.csv"):
    """
    Plots the main training progress and performance metrics from the log file.
    """
    script = os.path.dirname(__file__)
    save_path = os.path.join(script, "Plots", "Training and Performance")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
        
    df = pd.read_csv(log_file).dropna()

    # Plot 1: Training Metrics
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Training Progress Metrics', fontsize=16)

    axs1[0, 0].plot(df['timesteps'], df['total_reward'])
    axs1[0, 0].set_title('Total Reward vs. Training Duration')
    axs1[0, 0].set_xlabel('Timesteps')
    axs1[0, 0].set_ylabel('Total Reward')

    axs1[0, 1].plot(df['timesteps'], df['kl_divergence'])
    axs1[0, 1].set_title('KL Divergence vs. Training Duration')
    axs1[0, 1].set_xlabel('Timesteps')
    axs1[0, 1].set_ylabel('KL Divergence')

    axs1[1, 0].plot(df['timesteps'], df['clip_fraction'])
    axs1[1, 0].set_title('Clipped Policies vs. Training Duration')
    axs1[1, 0].set_xlabel('Timesteps')
    axs1[1, 0].set_ylabel('Clip Fraction')

    axs1[1, 1].plot(df['timesteps'], df['entropy_loss'])
    axs1[1, 1].set_title('Entropy Loss vs. Training Duration')
    axs1[1, 1].set_xlabel('Timesteps')
    axs1[1, 1].set_ylabel('Entropy Loss')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(os.path.join(save_path, "training_progress_metrics.png"))
    plt.show()

    # Plot 2: Step-wise Performance
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Step-wise Performance Metrics', fontsize=16)

    axs2[0, 0].plot(df['timesteps'], df['step_reward'])
    axs2[0, 0].set_title('Reward by Step')
    axs2[0, 0].set_xlabel('Timesteps')
    axs2[0, 0].set_ylabel('Reward')

    axs2[0, 1].plot(df['timesteps'], df['cost_of_intervention'])
    axs2[0, 1].set_title('Cost of Intervention by Step')
    axs2[0, 1].set_xlabel('Timesteps')
    axs2[0, 1].set_ylabel('Cost')

    axs2[1, 0].plot(df['timesteps'], df['pressure_deficit'])
    axs2[1, 0].set_title('Pressure Deficit by Step')
    axs2[1, 0].set_xlabel('Timesteps')
    axs2[1, 0].set_ylabel('Pressure Deficit')

    axs2[1, 1].plot(df['timesteps'], df['demand_satisfaction'])
    axs2[1, 1].set_title('Demand Satisfaction by Step')
    axs2[1, 1].set_xlabel('Timesteps')
    axs2[1, 1].set_ylabel('Demand Satisfaction (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(os.path.join(save_path, "stepwise_performance_metrics.png"))
    plt.show()

    return [fig1, fig2]

def plot_action_analysis(log_file="training_log.csv"):
    """
    Plots the agent's action analysis metrics from the log file.
    Note: Upgraded pipes are not plotted as the environment logic prevents them.
    """

    script = os.path.dirname(__file__)
    save_path = os.path.join(script, "Plots", "Action Analysis")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # if not os.path.exists(log_file):
    #     print(f"Log file not found: {log_file}")
    #     return

    df = pd.read_csv(log_file).dropna()
    
    fig = plt.figure(figsize=(12, 8))
    plt.title('Action Analysis Metrics vs. Timesteps', fontsize=16)
    
    plt.plot(df['timesteps'], df['step_reward'], label='Reward by Step', alpha=0.7)
    plt.plot(df['timesteps'], df['pipe_changes'], label='Frequency of Pipe Changes', alpha=0.7)
    plt.plot(df['timesteps'], df['downgraded_pipes'], label='Frequency of Downgraded Pipes', alpha=0.7)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path, "action_analysis_metrics.png"))
    plt.show()

    return fig

def plot_final_agent_rewards_by_scenario(scenario_rewards: dict):
    """
    Plots the final average reward for each scenario.
    
    Args:
        scenario_rewards: A dictionary with scenario names as keys and average rewards as values.
    """

    script = os.path.dirname(__file__)
    save_path = os.path.join(script, "Plots", "Reward_by_Scenario")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scenarios = list(scenario_rewards.keys())
    rewards = list(scenario_rewards.values())
    
    fig = plt.figure(figsize=(15, 8))
    plt.bar(scenarios, rewards, color='skyblue')
    
    plt.xlabel('Scenario')
    plt.ylabel('Average Reward')
    plt.title('Final Agent Reward by Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path,"final_agent_rewards_by_scenario.png"))
    plt.show()

    return fig