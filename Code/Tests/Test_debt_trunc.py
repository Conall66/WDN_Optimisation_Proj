
import os
import sys
import numpy as np

script = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(script)
if parent not in sys.path:
    sys.path.append(parent)

from PPO_Environment2 import WNTRGymEnv

# --- 1. Define Configurations for the Test ---

# Use the same base configs from your training script
PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}
NETWORK_CONFIG = {'max_nodes': 150, 'max_pipes': 200}
REWARD_CONFIG = {'mode': 'custom_normalized', 'max_cost_normalization': 1000000.0}
SCENARIOS = ['hanoi_sprawling_1'] # Use a single, predictable scenario

# --- 2. Create a Special Budget Config with a LOW Debt Limit ---

# We set a very low max_debt to make it easy to exceed.
# Start with a positive budget but allow it to be depleted quickly.
# BUDGET_CONFIG_TEST = {
#     "start_of_episode_budget": 50000.0,
#     "initial_budget_per_step": 0, # No extra money per step
#     "max_debt": 10000.0, # A very small debt limit!
#     "labour_cost_per_meter": 100.0
# }

BUDGET_CONFIG_HANOI = {
    "initial_budget_per_step": 500_000.0,
    "start_of_episode_budget": 10_000_000.0,
    "ongoing_debt_penalty_factor": 0.0001,
    "max_debt": 1_000_000.0,
    "labour_cost_per_meter": 100.0
}

print("--- Starting Debt Truncation Test ---")

# --- 3. Initialize the Environment ---

try:
    env = WNTRGymEnv(
        pipes_config=PIPES_CONFIG,
        scenarios=SCENARIOS,
        network_config=NETWORK_CONFIG,
        budget_config=BUDGET_CONFIG_HANOI, # Use our special test config
        reward_config=REWARD_CONFIG
    )
    print("Environment initialized successfully.")
    print(f"Test Parameters: Start Budget=${BUDGET_CONFIG_HANOI['start_of_episode_budget']:.2f}, Max Debt=${BUDGET_CONFIG_HANOI['max_debt']:.2f}")

    # --- 4. Run the Test Episode ---

    obs, info = env.reset()
    
    # We will force the agent to take the most expensive action possible:
    # Action index 6 corresponds to the largest diameter (1.0160m).
    expensive_action = 6
    
    # Loop for a maximum number of steps to prevent an infinite loop
    for step_num in range(100): 
        print(f"\n--- Step {step_num + 1} ---")
        
        # Take the predetermined expensive action
        obs, reward, terminated, truncated, info = env.step(expensive_action)
        
        # Get information for logging
        cost = info.get('cost_of_intervention', 0)
        budget = env.cumulative_budget
        
        print(f"Action Taken: {expensive_action} (Upgrade to largest diameter)")
        print(f"Cost of this intervention: ${cost:.2f}")
        print(f"Cumulative Budget is now: ${budget:.2f}")
        print(f"Episode State: terminated={terminated}, truncated={truncated}")
        
        # --- 5. Check for Truncation ---
        
        if truncated:
            print("\n✅ SUCCESS: Episode was truncated as expected.")
            if budget < -BUDGET_CONFIG_HANOI['max_debt']:
                print(f"Confirmation: Budget (${budget:.2f}) correctly exceeded the max debt limit ($-{BUDGET_CONFIG_HANOI['max_debt']:.2f}).")
            else:
                print(f"⚠️ WARNING: Episode truncated, but budget (${budget:.2f}) did not seem to exceed the max debt limit ($-{BUDGET_CONFIG_HANOI['max_debt']:.2f}).")
            break # Exit the loop as the test is complete
            
        if terminated:
            print("\n❌ FAILURE: Episode terminated normally before truncation could be tested.")
            break

    # If the loop finishes without truncation
    else:

        print("\n❌ FAILURE: Episode did not truncate after 100 steps.")

    env.close()

except Exception as e:
    print(f"\nAn error occurred during the test: {e}")