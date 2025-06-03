import os
import time
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Import necessary classes from your project files
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent
from Plot_Agents import PlottingCallback

# --- 1. CONFIGURATION ---

# Shared configuration for all training stages
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}
SCENARIOS = [
    'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
    'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
]
NUM_CPU = max(1, torch.multiprocessing.cpu_count() - 1)

# --- 2. CURRICULUM DEFINITION ---
CURRICULUM = [
    {
        "name": "Stage1_Minimize_Pressure_Deficit",
        "reward_mode": "minimize_pd",
        "timesteps": 50000,
        "ppo_config": {"learning_rate": 5e-4, "n_steps": 2048, "batch_size": 64, "gamma": 0.8}
    },
    {
        "name": "Stage2_Balance_PD_and_Cost",
        "reward_mode": "pd_and_cost",
        "timesteps": 75000,
        "ppo_config": {"learning_rate": 2e-4, "n_steps": 2048, "batch_size": 64, "gamma": 0.85}
    },
    {
        "name": "Stage3_Full_Objective_Fine_Tuning",
        "reward_mode": "full_objective",
        "timesteps": 50000,
        "ppo_config": {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 128, "gamma": 0.9}
    }
]

# --- 3. HELPER AND EVALUATION FUNCTIONS ---

def make_env(reward_mode: str):
    """Helper function to create a WNTRGymEnv instance with a specific reward mode."""
    def _init():
        # This function is used by SubprocVecEnv to create an environment in a separate process
        env = WNTRGymEnv(
            pipes=PIPES_CONFIG,
            scenarios=SCENARIOS,
            reward_mode=reward_mode
        )
        return env
    return _init

def evaluate_agent_by_scenario(model_path: str, reward_mode: str, num_episodes=3):
    """
    Evaluates a trained DRL agent on each scenario using the specified reward mode.
    """
    print(f"\nEvaluating DRL agent from {os.path.basename(model_path)}...")
    print(f"Evaluation Reward Mode: {reward_mode}")

    # For evaluation, a single dummy environment is sufficient and easier to manage.
    # The lambda function creates an environment factory.
    eval_env = DummyVecEnv([lambda: WNTRGymEnv(pipes=PIPES_CONFIG, scenarios=SCENARIOS, reward_mode=reward_mode)])

    # The agent's policy is loaded from the specified path.
    agent = GraphPPOAgent(eval_env, PIPES_CONFIG)
    agent.load(model_path)

    scenario_rewards = {}
    for scenario in SCENARIOS:
        print(f"  - Evaluating scenario: {scenario}")
        episode_rewards = []
        for _ in range(num_episodes):
            # Reset the environment to the specific scenario.
            # `env_method` calls a method on the underlying environment within the VecEnv.
            reset_output = eval_env.env_method("reset", scenario_name=scenario)
            obs = reset_output[0][0] # Extract the actual observation
            
            # The agent expects a batched observation, so we add a dimension.
            obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

            total_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward[0] # Reward from VecEnv is an array
            episode_rewards.append(total_reward)
        
        scenario_rewards[scenario] = np.mean(episode_rewards)

    eval_env.close()
    return scenario_rewards

def evaluate_random_policy_by_scenario(reward_mode: str, num_episodes=3):
    """
    Evaluates a random action policy as a baseline, using the specified reward mode.
    """
    print("\nEvaluating Random Policy baseline...")
    print(f"Evaluation Reward Mode: {reward_mode}")

    # A non-vectorized environment is used for the random policy evaluation.
    eval_env = WNTRGymEnv(pipes=PIPES_CONFIG, scenarios=SCENARIOS, reward_mode=reward_mode)
    
    scenario_rewards = {}
    for scenario in SCENARIOS:
        print(f"  - Evaluating scenario: {scenario}")
        episode_rewards = []
        for _ in range(num_episodes):
            obs, _ = eval_env.reset(scenario_name=scenario)
            total_reward = 0
            done = False
            while not done:
                action = eval_env.action_space.sample() # Take a random action
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)
        
        scenario_rewards[scenario] = np.mean(episode_rewards)

    eval_env.close()
    return scenario_rewards

def plot_scenario_comparison(drl_rewards: dict, random_rewards: dict, stage_name: str, save_path: str):
    """
    Generates and saves a bar chart comparing DRL agent rewards to a random policy.
    This function is adapted from `plot_final_agent_rewards_by_scenario`.
    """
    scenarios = list(drl_rewards.keys())
    drl_scores = [drl_rewards.get(s, 0) for s in scenarios]
    random_scores = [random_rewards.get(s, 0) for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 9))
    rects1 = ax.bar(x - width/2, drl_scores, width, label='DRL Agent', color='deepskyblue')
    rects2 = ax.bar(x + width/2, random_scores, width, label='Random Policy', color='lightcoral')

    ax.set_ylabel('Average Reward')
    ax.set_title(f'End-of-Stage Evaluation: {stage_name}\nDRL Agent vs. Random Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Saved evaluation plot to: {save_path}")
    plt.close(fig) # Close the figure to free up memory


# --- 4. MAIN TRAINING AND EVALUATION SCRIPT ---

def main():
    """
    Main function to run the curriculum learning process, including evaluation.
    """
    print("--- Starting Curriculum Learning Protocol with Evaluation ---")
    
    # Create directories for saving models, logs, and plots
    script_dir = os.path.dirname(__file__)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"curriculum_run_{run_timestamp}"
    
    agents_dir = os.path.join(script_dir, "agents", run_dir_name)
    logs_dir = os.path.join(script_dir, "Plots", run_dir_name, "training_logs")
    eval_plots_dir = os.path.join(script_dir, "Plots", run_dir_name, "evaluation_charts")
    
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)

    previous_model_path = None

    # --- Training Loop ---
    for i, stage in enumerate(CURRICULUM):
        stage_name = stage["name"]
        reward_mode = stage["reward_mode"]
        timesteps = stage["timesteps"]
        ppo_config = stage["ppo_config"]
        
        print("\n" + "="*80)
        print(f"  STARTING: {stage_name} (Stage {i+1}/{len(CURRICULUM)})")
        print(f"  Reward Mode: {reward_mode} | Timesteps: {timesteps}")
        print("="*80)

        vec_env = SubprocVecEnv([make_env(reward_mode) for _ in range(NUM_CPU)])
        agent = GraphPPOAgent(vec_env, PIPES_CONFIG, **ppo_config)

        if previous_model_path:
            print(f"Loading model from previous stage: {os.path.basename(previous_model_path)}")
            agent.agent.set_env(vec_env) 
            agent.load(previous_model_path)
        else:
            print("Initializing a new agent for the first stage.")

        plotting_callback = PlottingCallback(verbose=1)

        start_time = time.time()
        agent.train(total_timesteps=timesteps, callback=plotting_callback)
        training_time = time.time() - start_time
        print(f"\nStage '{stage_name}' training completed in {training_time:.2f} seconds.")

        # Save the model and logs
        model_filename = f"{i+1}_{stage_name}"
        current_model_path = os.path.join(agents_dir, model_filename)
        agent.save(current_model_path)
        print(f"Model for this stage saved to: {current_model_path}.zip")

        log_df = pd.DataFrame(plotting_callback.log_data)
        if not log_df.empty:
            log_path = os.path.join(logs_dir, f"{model_filename}_log.csv")
            log_df.to_csv(log_path, index=False)
            print(f"Training log for this stage saved to: {log_path}")

        vec_env.close()

        # --- NEW: Evaluation and Plotting Step ---
        print("\n--- Performing End-of-Stage Evaluation ---")
        
        # Evaluate the DRL agent that was just trained
        drl_results = evaluate_agent_by_scenario(
            model_path=current_model_path,
            reward_mode=reward_mode
        )
        
        # Evaluate the random policy using the same reward function for a fair comparison
        random_results = evaluate_random_policy_by_scenario(
            reward_mode=reward_mode
        )
        
        # Plot the comparison
        plot_save_path = os.path.join(eval_plots_dir, f"{model_filename}_comparison.png")
        plot_scenario_comparison(drl_results, random_results, stage_name, plot_save_path)
        
        # Set up for the next stage
        previous_model_path = current_model_path

    print("\n--- Curriculum Learning Protocol Finished ---")

if __name__ == "__main__":

    # This is important for multiprocessing on some OS.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()