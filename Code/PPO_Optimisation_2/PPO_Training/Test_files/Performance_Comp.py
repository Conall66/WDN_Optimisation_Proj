import time
import torch
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import sys
import os

script = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import your existing modules
from PPO_Environment import WNTRGymEnv
from Actor_Critic_Nets2 import GraphPPOAgent

def run_training_experiment(
    use_vectorized: bool,
    num_envs: int,
    device: str,
    total_timesteps: int = 10000 # Small timesteps for quick testing and comparison
    ):
    """
    Runs a single training experiment with a specific configuration.

    Args:
        use_vectorized: Whether to use a vectorized environment.
        num_envs: The number of parallel environments to use (if vectorized).
        device: The torch device to use ('cpu' or 'cuda').
        total_timesteps: The number of timesteps to train for.

    Returns:
        The time taken for the training run in seconds.
    """
    
    # --- Configuration ---
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    scenarios = ['anytown_densifying_1', 'hanoi_densifying_1'] # Using fewer scenarios for a quicker test

    # --- Environment Setup ---
    def make_env():
        return WNTRGymEnv(pipes, scenarios)

    if use_vectorized:
        # Use SubprocVecEnv for multi-processing, DummyVecEnv for single-process vectorization
        vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
        env = vec_env_cls([make_env for _ in range(num_envs)])
    else:
        # A single, non-vectorized environment
        env = make_env()

    # --- Agent Setup ---
    ppo_config = {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64}
    agent = GraphPPOAgent(env, pipes, device=device, **ppo_config)

    # --- Run Training and Time it ---
    start_time = time.time()
    agent.train(total_timesteps=total_timesteps)
    end_time = time.time()

    env.close()
    return end_time - start_time

def plot_performance_comparison(results: dict, save_path):
    """
    Plots the training duration comparison as a bar chart.
    """
    labels = list(results.keys())
    durations = list(results.values())

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, durations, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    ax.set_ylabel('Training Duration (seconds)')
    ax.set_title('Training Performance Comparison')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")

    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_performance_comparison.png"))
    plt.show()

if __name__ == "__main__":
    # --- Experiment Configurations ---
    results = {}
    cpu_count = mp.cpu_count()
    gpu_available = torch.cuda.is_available()
    
    print("Starting performance comparison experiments...")

    # 1. Serial CPU (1 environment on 1 CPU core)
    print("\nRunning: Serial CPU...")
    duration = run_training_experiment(use_vectorized=False, num_envs=1, device='cpu')
    results['Serial CPU (1 Env)'] = duration
    print(f"Completed in {duration:.2f} seconds.")

    # 2. Parallel CPU (Multiple environments on multiple CPU cores)
    print(f"\nRunning: Parallel CPU ({cpu_count} Envs)...")
    duration = run_training_experiment(use_vectorized=True, num_envs=cpu_count, device='cpu')
    results[f'Parallel CPU ({cpu_count} Envs)'] = duration
    print(f"Completed in {duration:.2f} seconds.")

    # 3. GPU experiments (if available)
    if gpu_available:
        # 3a. Serial on GPU (1 environment, model on GPU)
        print("\nRunning: Serial GPU (1 Env)...")
        duration = run_training_experiment(use_vectorized=False, num_envs=1, device='cuda')
        results['Serial GPU (1 Env)'] = duration
        print(f"Completed in {duration:.2f} seconds.")

        # 3b. Parallel on GPU (Multiple envs on CPU, model on GPU)
        print(f"\nRunning: Parallel GPU ({cpu_count} Envs)...")
        duration = run_training_experiment(use_vectorized=True, num_envs=cpu_count, device='cuda')
        results[f'Parallel GPU ({cpu_count} Envs)'] = duration
        print(f"Completed in {duration:.2f} seconds.")
    else:
        print("\nSkipping GPU experiments: CUDA is not available.")

    # --- Plot the results ---
    print("\nAll experiments complete. Plotting results...")
    save_path = os.path.join(script, "Plots", "Performance_Comparison")
    os.makedirs(save_path, exist_ok=True)
    plot_performance_comparison(results, save_path)