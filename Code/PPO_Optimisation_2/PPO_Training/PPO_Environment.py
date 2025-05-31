"""
In this file, we take the .inp files from the Modified_nets folder and create a gym environment for each network.

This version is corrected to output graph-based dictionary observations suitable for the GNN,
which resolves the ValueError from changing observation sizes.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wntr
import os
import random
from typing import Dict, List, Tuple, Optional
import networkx as nx

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import calculate_reward

class WNTRGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            pipes: Dict,
            scenarios: List[str],
            networks_folder: str = 'Modified_nets',
            # Define max network size for padding. Adjust if your networks can be larger.
            max_nodes: int = 150,
            max_pipes: int = 200,
            ):
        
        super(WNTRGymEnv, self).__init__()

        self.pipes = pipes
        self.pipe_diameter_options = [p['diameter'] for p in pipes.values()]
        self.scenarios = scenarios
        self.networks_folder = networks_folder
        self.labour_cost = 100

        # --- New parameters for handling variable graph sizes ---
        self.max_nodes = max_nodes
        self.max_pipes = max_pipes
        
        self.current_scenario = None
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.network_states = {}
        self.current_network = None
        self.pipe_names = []
        self.node_names = []
        self.node_pressures = {}
        self.actions_this_timestep = []
        self.original_diameters_this_timestep = {}
        
        # --- Action Space (Unchanged) ---
        self.action_space = spaces.Discrete(len(self.pipe_diameter_options) + 1)
        
        # --- Observation Space is now a Dictionary ---
        self.observation_space = spaces.Dict({
            # Node features: [demand, elevation, pressure, is_junction]
            "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, 4), dtype=np.float32),
            # Edge features: [diameter, length, roughness, is_current_pipe]
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_pipes, 4), dtype=np.float32),
            # Global features: [num_nodes, num_pipes, current_pipe_index]
            "globals": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

    def load_network_states(self, scenario: str) -> Dict[int, str]:
        scenario_path = os.path.join(self.networks_folder, scenario)
        inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])
        return {i: os.path.join(scenario_path, f) for i, f in enumerate(inp_files)}

    def get_network_features(self) -> Dict[str, np.ndarray]:
        """
        Extracts features as a dictionary of padded numpy arrays, suitable for the GNN.
        """
        # --- Padded arrays for features ---
        node_features = np.zeros((self.max_nodes, 4), dtype=np.float32)
        edge_features = np.zeros((self.max_pipes, 4), dtype=np.float32)

        # --- Extract Node Features ---
        num_nodes = len(list(self.current_network.junctions()))
        for i, (node_name, node) in enumerate(self.current_network.junctions()):
            if i < self.max_nodes:
                node_features[i] = [
                    node.base_demand or 0.0,
                    node.elevation or 0.0,
                    self.node_pressures.get(node_name, 0.0),
                    1.0  # is_junction
                ]

        # --- Extract Edge (Pipe) Features ---
        num_pipes = len(self.pipe_names)
        for i, pipe_name in enumerate(self.pipe_names):
             if i < self.max_pipes:
                pipe = self.current_network.get_link(pipe_name)
                is_current_pipe = 1.0 if i == self.current_pipe_index else 0.0
                edge_features[i] = [
                    pipe.diameter,
                    pipe.length,
                    pipe.roughness,
                    is_current_pipe
                ]

        # --- Extract Global Features ---
        global_features = np.array([
            num_nodes,
            num_pipes,
            self.current_pipe_index
        ], dtype=np.float32)

        return {"nodes": node_features, "edges": edge_features, "globals": global_features}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        self.current_scenario = self.np_random.choice(self.scenarios)
        self.network_states = self.load_network_states(self.current_scenario)
        
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.actions_this_timestep = []
        
        self._load_network_for_timestep()
        
        obs = self.get_network_features()
        info = {'scenario': self.current_scenario, 'time_step': 0, 'pipe_index': 0}
        
        return obs, info

    def _load_network_for_timestep(self):
        """Helper function to load a network and set its initial state."""
        network_path = self.network_states[self.current_time_step]
        self.current_network = wntr.network.WaterNetworkModel(network_path)
        self.pipe_names = self.current_network.pipe_name_list
        self.node_names = self.current_network.junction_name_list

        self.original_diameters_this_timestep = {p: self.current_network.get_link(p).diameter for p in self.pipe_names}
        
        self.node_pressures = {}
        results, _ = self.simulate_network(self.current_network)
        if results:
            # Important: Sanitize the pressure results
            pressures = results.node['pressure']
            # Convert NaN to 0, inf to large numbers
            # sanitized_pressures = np.nan_to_num(pressures, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if hasattr(pressures, 'loc'):
                for node_name in self.node_names:
                    # Get all pressure values for this node across the simulation time
                    node_pressures = pressures.loc[:, node_name].values
                    
                    # FIX: Use np.nanmean to safely calculate the average, ignoring NaNs.
                    # If all values are NaN, nanmean returns NaN, so we check for that.
                    mean_pressure = np.nanmean(node_pressures)
                    
                    # If the result is still NaN (e.g., all pressures were NaN), default to 0.
                    self.node_pressures[node_name] = 0.0 if np.isnan(mean_pressure) else mean_pressure

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        terminated = False
        truncated = False
        
        # Default info and reward for an intermediate step (acting on a single pipe)
        info = {}
        reward = 0.0

        # --- Logic for taking an action on a single pipe ---
        pipe_name = self.pipe_names[self.current_pipe_index]
        old_diameter = self.current_network.get_link(pipe_name).diameter

        # Action 0 is "do nothing"
        if action > 0:
            new_diameter = self.pipe_diameter_options[action - 1]
            # MODIFIED: Only allow upgrades
            if new_diameter > old_diameter: 
                self.current_network.get_link(pipe_name).diameter = new_diameter
                self.actions_this_timestep.append((pipe_name, new_diameter))

        self.current_pipe_index += 1

        # --- Logic for when all pipes have been decided for the current network state ---
        if self.current_pipe_index >= len(self.pipe_names):
            # First, simulate the network with the agent's chosen actions
            results, metrics = self.simulate_network(self.current_network)
            
            if results:
                # --- MODIFICATION START: Calculate max_pd for reward normalization ---
                max_pd = 0.0
                try:
                    # Create a temporary copy of the network to find the worst-case pressure deficit
                    network_path = self.network_states[self.current_time_step]
                    wn_copy = wntr.network.WaterNetworkModel(network_path)
                    
                    # Get the smallest available pipe diameter
                    min_diameter = min(p['diameter'] for p in self.pipes.values())

                    # Set all pipes in the temporary network to the smallest diameter
                    for p_name in wn_copy.pipe_name_list:
                        wn_copy.get_link(p_name).diameter = min_diameter
                    
                    # Simulate this "worst-case" network
                    max_pd_results, _ = self.simulate_network(wn_copy)
                    if max_pd_results:
                        max_pd_metrics = evaluate_network_performance(wn_copy, max_pd_results)
                        max_pd = max_pd_metrics.get('total_pressure_deficit', 0.0)

                except Exception as e:
                    print(f"Error calculating max_pd: {e}")
                # --- MODIFICATION END ---

                downgraded = False # Downgrades are prevented by the action mask
                disconnected, bad_actions = (False, []) # Placeholder

                # MODIFICATION: Pass the calculated max_pd to the reward function
                reward_tuple = calculate_reward(
                    self.current_network, self.original_diameters_this_timestep, 
                    self.actions_this_timestep, self.pipes, metrics, 
                    self.labour_cost, downgraded, disconnected, bad_actions,
                    max_pd=max_pd
                )
                reward = reward_tuple[0]
                
                # This is the data your plotting callback will log.
                info = {
                    'reward': reward,
                    'cost_of_intervention': reward_tuple[1],
                    'pressure_deficit': reward_tuple[2],
                    'demand_satisfaction': reward_tuple[3],
                    'pipe_changes': len(self.actions_this_timestep),
                    'upgraded_pipes': len(self.actions_this_timestep),
                }

            else: # Main simulation failed
                reward = -1000.0
                info = {} # Return empty info on failure
            
            # --- Reset for the next major timestep in the scenario ---
            self.current_time_step += 1
            self.current_pipe_index = 0
            self.actions_this_timestep = []

            if self.current_time_step >= len(self.network_states) or not results:
                terminated = True # End of episode
            else:
                self._load_network_for_timestep() # Load the next network state
        
        # Get the observation for the current state (or next state)
        obs = self.get_network_features()

        return obs, reward, terminated, truncated, info

    def simulate_network(self, network: wntr.network.WaterNetworkModel):
        try:
            results = run_epanet_simulation(network)
            # Check for NaNs in the results right after simulation
            if results.node['pressure'].isnull().values.any():
                print(f"WARNING: NaN pressure values detected in scenario: {self.current_scenario}, time_step: {self.current_time_step}")
            metrics = evaluate_network_performance(network, results)
            return results, metrics
        except Exception as e:
            # Log the error with context
            print(f"ERROR: Simulation failed for scenario: {self.current_scenario}, time_step: {self.current_time_step}. Error: {e}")
            return None, None
            
    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=bool)
        current_diameter = self.current_network.get_link(self.pipe_names[self.current_pipe_index]).diameter
        
        # Action 0 is "do nothing", which we always allow.
        mask[0] = True
        
        # Iterate through the diameter options to determine valid upgrades.
        for i, new_diameter in enumerate(self.pipe_diameter_options):
            # The action is valid only if the new diameter is an upgrade.
            if new_diameter > current_diameter:
                mask[i + 1] = True  # Allow this action
            else:
                mask[i + 1] = False # Disallow this action
                
        return mask

    def close(self):
        pass