
"""

In this script, we take load a set of .inp files and create a gym environment for reinforcement learning. The environment simulates a water distribution network and allows an agent to interact with it by modifying the pipe diameters. The agent will take an action for each pipe in the network, and subsequently these changes will be translated into the next network model to repeat. Each new network builds topoligically on the previous network.

"""

import wntr
from wntr.network import WaterNetworkModel
import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
import torch
from torch_geometric.data import Data

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance

from Hydraulic import run_epanet_sim

class ObjectSpace(spaces.Space):
    """
    A custom Gymnasium space for holding arbitrary Python objects.
    This is necessary because the default `DummyVecEnv` in Stable-Baselines3
    cannot handle variable-sized spaces like `spaces.Graph` directly.
    By defining a space that can hold objects, we allow the VecEnv to create
    a buffer of objects, bypassing the shape-checking errors.
    """
    def __init__(self):
        super().__init__(shape=(), dtype=object)

    def sample(self):
        # Sampling is not well-defined for arbitrary objects.
        # This is not used by the PPO agent during training.
        raise NotImplementedError("Sampling not supported for this space")

    def contains(self, x) -> bool:
        # We can check if x is a GraphInstance, but for simplicity,
        # we'll just accept any object.
        return True

class PPOEnv(gym.Env):
    """
    Custom Gym environment for simulating a water distribution network using the WNTR library.
    """

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 30
    }

    def __init__(
            self,
            network_files_dir='Networks/Simple_Nets',
            ):
        """
        Initialise the environment with a water network model loaded from an .inp file.
        """
        super(PPOEnv, self).__init__()

        # --- File management and network setup ---
        self.network_files_dir = network_files_dir
        self.network_files = sorted(
            [f for f in os.listdir(self.network_files_dir) if f.endswith('.inp')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        self.initial_wn = WaterNetworkModel(os.path.join(self.network_files_dir, self.network_files[0]))
        self.wn = None

        # --- Environment parameters ---
        self.initial_budget = 500000.0 # Stored for normalization
        self.budget = self.initial_budget
        self.budget_step = 200000.0
        self.energy_price_kwh = 0.26
        self.labour_cost = 100.0
        self.min_pressure_m = 10.0
        self.pipes = {
            'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
            'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
            'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
            'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
            'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
            'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
        }

        # --- State tracking ---
        self.current_step_index = 0
        self.current_pipe_index = 0
        self.total_cost = 0.0
        self.results = None

        self.action_space = spaces.Discrete(len(self.pipes) + 1)

        self.observation_space = spaces.Dict({
            "graph": ObjectSpace(),
            "global_state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "current_pipe_nodes": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.int64)
        })

    def _get_observation(self):
        # --- Define reasonable maximums for normalisation ---
        max_pressure = 100
        max_demand = 0.1
        max_diameter = max(self.pipes[pipe]['diameter'] for pipe in self.pipes)
        max_length = 5000

        # --- Extract raw data and create node mapping ---
        pressures = self.results.node['pressure'].iloc[-1]
        demands = self.results.node['demand'].iloc[-1]
        node_map = {name: i for i, name in enumerate(self.wn.node_name_list)}

        # --- Create and normalise Node Features ---
        raw_node_features = np.zeros((len(node_map), 2), dtype=np.float32)
        for name, i in node_map.items():
            raw_node_features[i, 0] = pressures.get(name, 0)
            raw_node_features[i, 1] = demands.get(name, 0)
        norm_node_features = np.clip(raw_node_features / np.array([max_pressure, max_demand]), 0, 1)

        # --- Create and normalise Edge Features ---
        start_nodes, end_nodes = [], []
        raw_edge_features = []
        current_pipe_name = self.wn.pipe_name_list[self.current_pipe_index]
        for pipe_name, pipe in self.wn.pipes():
            start_nodes.extend([node_map[pipe.start_node_name], node_map[pipe.end_node_name]])
            end_nodes.extend([node_map[pipe.end_node_name], node_map[pipe.start_node_name]])
            is_current = 1.0 if pipe_name == current_pipe_name else 0.0
            features = [pipe.diameter, pipe.length, is_current]
            raw_edge_features.extend([features, features])
        edge_index = np.array([start_nodes, end_nodes], dtype=np.int64)
        norm_edge_features = np.clip(np.array(raw_edge_features) / np.array([max_diameter, max_length, 1.0]), 0, 1)

        # --- Create Global State Vector ---
        norm_budget = self.budget / self.initial_budget
        norm_total_cost = self.total_cost / self.initial_budget
        # Avoid division by zero if there's only one network
        total_networks = len(self.network_files)
        norm_network_index = self.current_step_index / (total_networks - 1) if total_networks > 1 else 0
        global_state = np.array([norm_budget, norm_total_cost, norm_network_index], dtype=np.float32)
        global_state = np.clip(global_state, 0, 1)

        # --- Get Current Pipe Node Indices ---
        pipe_obj = self.wn.get_link(current_pipe_name)
        start_node_idx = node_map[pipe_obj.start_node_name]
        end_node_idx = node_map[pipe_obj.end_node_name]
        current_pipe_nodes = np.array([start_node_idx, end_node_idx], dtype=np.int64)

        edge_links = edge_index.T

        # Create the observation dictionary with the new structure
        observation = {
            "graph": GraphInstance(
                nodes=norm_node_features.astype(np.float32),
                edges=norm_edge_features.astype(np.float32),
                edge_links=edge_links.astype(np.int64)
            ),
            "global_state": global_state.astype(np.float32),
            "current_pipe_nodes": current_pipe_nodes.astype(np.int64)
        }
        return observation

    def _calculate_reward(self, weights=None):
        """
        Calculates a reward score between 0 and 1.
        
        This function balances hydraulic performance, energy consumption, and
        budget utilization, with each component being normalised to a [0,1] score.
        """
        # --- 1. Pressure Score (0 to 1, where 1 is best) ---
        pressures = self.results.node['pressure'].iloc[-1]
        pressure_deficit = np.sum(np.maximum(0, self.min_pressure_m - pressures.loc[self.wn.junction_name_list]))
        
        # Define a worst-case scenario for normalisation
        max_possible_deficit = self.min_pressure_m * len(self.wn.junction_name_list)
        # print(f"Maximum possible pressure deficit: {max_possible_deficit:.2f} m")

        pressure_score = max(0, 1 - (pressure_deficit / max_possible_deficit)) if max_possible_deficit > 0 else 1.0

        # Cost reward is normalised to [0,1] scale, with 1 bein gno mony spent, and 0 being the budget exhausted

        normalised_cost = self.total_cost / self.initial_budget
        cost_score = max(0, 1 - normalised_cost) if self.initial_budget > 0 else 1.0

        # --- 4. Final Weighted Combination ---
        # Define the importance of each component
        if weights == None:
            weights = {
                'pressure deficit': 0.4,
                'cost': 0.6,
                # 'energy': 0.2
            }

        weighted_pressure = weights['pressure deficit'] * pressure_score
        weighted_cost = weights['cost'] * cost_score
        
        total_reward = weighted_pressure + weighted_cost

        return total_reward, weighted_pressure, weighted_cost, pressure_deficit
    
    def reset(self, budget = 500000, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        Parameters:
        seed (int): Random seed for reproducibility.
        options (dict): Additional options for resetting the environment.
        Returns:
        observation (dict): Initial observation of the environment.
        """

        super().reset(seed=seed)
        self.budget = budget # Set a budget for the episode, this can be changed to a more dynamic value in the future
        """In future, move the budget out of the reset function and make it a parameter of the environment, so that it can be changed dynamically."""
        self.current_step_index = 0
        self.current_pipe_index = 0
        self.total_cost = 0.0
        
        # Load the first network file
        initial_inp_file = os.path.join(self.network_files_dir, self.network_files[0])
        self.wn = WaterNetworkModel(initial_inp_file)
        
        # Run initial simulation
        # sim = wntr.sim.EpanetSimulator(self.wn)
        # self.results = sim.run_sim()

        results = run_epanet_sim(self.wn) # Run the simulation using the custom function
        self.results = results

        return self._get_observation(), {"total_cost": self.total_cost} # Total cost is updated in each step, so we return it here for the first observation
    
    def step(self, action):
        """Executes one agent-environment interaction step."""

        self.step_cost = 0.0
        pipe_to_modify_name = self.wn.pipe_name_list[self.current_pipe_index]
        pipe_obj = self.wn.get_link(pipe_to_modify_name)
        
        # Apply the action
        if action == 0:
            # No change to the pipe diameter
            pass
        else:
            # Map the action to a new diameter
            new_diameter = list(self.pipes.values())[action - 1]['diameter']
            old_diameter = pipe_obj.diameter
                
            # Change cost calculation
            change_cost = (self.pipes[list(self.pipes.keys())[action - 1]]['unit_cost'] * pipe_obj.length + self.labour_cost * pipe_obj.length)

            # Remove step cost from budget
            self.budget -= change_cost
            self.step_cost += change_cost
            self.total_cost += change_cost
            pipe_obj.diameter = new_diameter
        
        # --- Transition to the next state ---
        self.current_pipe_index += 1
        terminated = False
        reward = 0  # Default reward is 0 for intermediate steps

        # Check if we've finished the current network
        network_completed = self.current_pipe_index >= len(self.wn.pipe_name_list)
        
        # If we've considered all pipes in the current network...
        if network_completed:
            self.current_step_index += 1
            
            # Run simulation to calculate reward for the completed network
            try:
                results = run_epanet_sim(self.wn)
                self.results = results
                # Calculate reward only after completing all pipes in a network
                reward, weighted_pressure, weighted_cost, pressure_deficit = self._calculate_reward()
            except Exception as e:
                print(f"Simulation failed at network completion: {e}")
                reward = -1  # Penalty for simulation failure
                terminated = True
                
            # ...check if there are more network files to load.
            if self.current_step_index >= len(self.network_files):
                terminated = True  # End of the episode
            else:
                # Increase the budget
                self.budget += self.budget_step
                # Load the next network, carrying over modified diameters
                next_inp_file = os.path.join(self.network_files_dir, self.network_files[self.current_step_index])
                next_wn = WaterNetworkModel(next_inp_file)
                for pipe_name, pipe in self.wn.pipes():
                    if pipe_name in next_wn.pipe_name_list:
                        next_wn.get_link(pipe_name).diameter = pipe.diameter
                self.wn = next_wn
                self.current_pipe_index = 0  # Reset pipe index for new network
        else:
            # For intermediate steps, just run simulation to update the state
            try:
                results = run_epanet_sim(self.wn)
                self.results = results
                # No reward calculation for intermediate steps
            except Exception as e:
                print(f"Simulation failed at step {self.current_step_index}, pipe {self.current_pipe_index}: {e}")
                reward = -1  # Penalty for simulation failure
                terminated = True

        observation = self._get_observation()
        info = {
            'total_cost': self.total_cost, 
            'step_cost': self.step_cost,
            'weighted_pressure': weighted_pressure if 'weighted_pressure' in locals() else 0,
            'weighted_cost': weighted_cost if 'weighted_cost' in locals() else 0,
            'pressure_deficit': pressure_deficit if 'pressure_deficit' in locals() else 0,
            'network_completed': network_completed,  # Flag to indicate if a network was completed
            'reward': reward  # Include reward in info for clarity
        }
        truncated = False  # Not using time limits

        return observation, reward, terminated, truncated, info

if __name__ == '__main__':

    test_dir = 'Networks/Simple_Nets'
    env = PPOEnv(network_files_dir=test_dir)
    
    initial_budget = 500000  # Store the initial budget separately
    obs, info = env.reset(budget=initial_budget)
    print("--- Environment Initialized ---")
    print(f"Initial network: {env.network_files[env.current_step_index]}")
    print(f"Number of pipes in the network: {len(env.wn.pipe_name_list)}")
    print(f"Budget: £{env.budget:,.2f}")
    print("-" * 80)
    
    current_network_index = env.current_step_index
    network_actions = []
    
    for i in range(15):
        # Store the current pipe's original diameter before action
        pipe_to_modify = env.wn.pipe_name_list[env.current_pipe_index]
        original_diameter = env.wn.get_link(pipe_to_modify).diameter
        
        # Take action
        action = env.action_space.sample() # Randomly sample actions from action space
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store action for network summary
        network_actions.append(action)
        
        # Get new diameter after action
        new_diameter = original_diameter  # Default if no change
        if action > 0:
            new_diameter = list(env.pipes.values())[action - 1]['diameter']
        
        # Translate action to human-readable description
        action_desc = "No change" if action == 0 else f"Change to {list(env.pipes.keys())[action - 1]}"
        
        # Print step information
        print(f"Step {i+1}: Pipe {pipe_to_modify}")
        print(f"  Action: {action} ({action_desc})")
        print(f"  Diameter: {original_diameter:.4f}m → {new_diameter:.4f}m")
        print(f"  Step Cost: £{info['step_cost']:,.2f}")
        # print(f"  Remaining Budget: £{env.budget:,.2f}")
        print(f"  Total Cost: £{info['total_cost']:,.2f}")
        
        # Print reward only if a network was completed
        if info.get('network_completed', False):
            print(f"  Network Completed! Reward: {reward:.4f}")
            print(f"    - Weighted Pressure: {info['weighted_pressure']:.4f}")
            print(f"    - Weighted Cost: {info['weighted_cost']:.4f}")
            print(f"    - Pressure Deficit: {info['pressure_deficit']:.4f} m")
        
        # Check if we've moved to a new network
        if env.current_step_index != current_network_index:
            print("\n" + "=" * 80)
            print(f"Network {env.network_files[current_network_index]} completed!")
            print(f"Reward for completed network: {reward:.4f}")
            
            # Store the budget before bonus for display
            budget_before_bonus = env.budget - env.budget_step
            
            # The budget_step has already been added in the step method,
            # so we're just displaying the values here
            print(f"Budget before bonus: £{budget_before_bonus:,.2f}")
            print(f"New budget: £{env.budget:,.2f}")  # This will now show the updated value
            print(f"Total cost so far: £{info['total_cost']:,.2f} ({info['total_cost']/initial_budget*100:.1f}% of initial budget)")
            print("=" * 80 + "\n")
            
            # Reset for next network
            current_network_index = env.current_step_index
            network_actions = []
            
            if not terminated:  # Only print if not terminated
                print(f"--- Loaded new network: {env.network_files[current_network_index]} ---")
                print(f"Number of pipes in new network: {len(env.wn.pipe_name_list)}")
                print("-" * 80)
        
        if terminated or truncated:
            print("\n--- Episode Finished ---")
            print(f"Final cost: £{info['total_cost']:,.2f} ({info['total_cost']/initial_budget*100:.1f}% of initial budget)")
            print(f"Remaining budget: £{env.budget:,.2f}")
            break
            
    env.close()

    
