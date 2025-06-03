import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import datetime

from Train_w_Plots import train_agent_with_monitoring, evaluate_agent_by_scenario

# --- Define the search space for hyperparameters ---
search_space = {
    "learning_rate": [1e-3, 5e-4, 2e-4],
    "gamma": [0.90, 0.95, 0.99],
    "ent_coef": [0.0, 0.01, 0.02],
    "n_steps": [2048],  # Can be expanded
    "batch_size": [64, 128]
}

# --- Create a list of all combinations ---
keys, values = zip(*search_space.items())
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

# --- Wrapper for a single experiment ---
def run_experiment(run_id, config):
    print(f"\n[Run {run_id}] Starting config: {config}")
    model_path, pipes, scenarios = train_agent_with_monitoring(
        net_type='hanoi',
        time_steps=50000
        # custom_config=config
    )

    rewards = evaluate_agent_by_scenario(model_path, pipes, scenarios, num_episodes_per_scenario=3)
    avg_reward = np.mean(list(rewards.values()))
    
    return {
        "run_id": run_id,
        **config,
        "avg_reward": avg_reward,
        "model_path": model_path
    }

# --- Run all experiments in parallel ---
def run_all_experiments():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        args = [(i, config) for i, config in enumerate(configs)]
        for result in pool.starmap(run_experiment, args):
            results.append(result)

    df = pd.DataFrame(results)
    save_path = f"hyperparam_results_{timestamp}.csv"
    df.to_csv(save_path, index=False)
    print(f"\nSaved results to {save_path}")
    return df

if __name__ == "__main__":
    run_all_experiments()