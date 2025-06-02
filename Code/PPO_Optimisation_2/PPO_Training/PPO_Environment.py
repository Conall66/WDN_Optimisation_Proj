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
from Reward import calculate_reward, reward_just_pd, compute_total_cost, reward_minimise_pd, reward_pd_and_cost, reward_full_objective

class WNTRGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            pipes: Dict,
            scenarios: List[str],
            networks_folder: str = 'Modified_nets',
            # Define max network size for padding. Adjust if your networks can be larger.
            max_nodes: int = 150,
            max_pipes: int = 200, # These parameters determine rollout buffer size
            current_max_pd = 0.0,
            current_max_cost = 0.0
            ):
        
        super(WNTRGymEnv, self).__init__()

        self.pipes = pipes
        self.pipe_diameter_options = [p['diameter'] for p in pipes.values()]
        self.scenarios = scenarios
        self.networks_folder = networks_folder
        self.labour_cost = 100

        self.max_nodes = max_nodes
        self.max_pipes = max_pipes
        self.current_max_pd = current_max_pd
        self.current_max_cost = current_max_cost
        # self.reward_mode = reward_mode
        
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
        self.chosen_diameters_from_previous_step = None
        
        # --- Action Space (Unchanged) ---
        self.action_space = spaces.Discrete(len(self.pipe_diameter_options) + 1)
        
        # --- Observation Space is now a Dictionary ---
        self.observation_space = spaces.Dict({
            # Node features: [demand, elevation, pressure, is_junction]
            "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, 4), dtype=np.float32),
            # Edge features: [diameter, length, roughness, is_current_pipe]
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_pipes, 4), dtype=np.float32),
            # Edge connectivity: Shape is [2, num_edges]. We use max_pipes * 2 for bidirectional edges.
            "edge_index": spaces.Box(low=0, high=max_nodes, shape=(2, max_pipes * 2), dtype=np.int32),
            # Global features: [num_nodes, num_pipes, current_pipe_index]
            "globals": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

            # ACTION MASK REMOVED
        })

    def load_network_states(self, scenario: str) -> Dict[int, str]:
        scenario_path = os.path.join(self.networks_folder, scenario)
        inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])
        return {i: os.path.join(scenario_path, f) for i, f in enumerate(inp_files)}

    def get_network_features(self) -> Dict[str, np.ndarray]:
        """
        Extracts features, including edge_index AND action_mask, 
        as a dictionary of padded numpy arrays.
        """
        # --- Padded arrays for features ---
        node_features = np.zeros((self.max_nodes, 4), dtype=np.float32)
        edge_features = np.zeros((self.max_pipes, 4), dtype=np.float32) # This size should be self.max_pipes * 2 if using bidirectional edge features
        # Or ensure GNNFeatureExtractor handles self.max_pipes if features are per unique pipe
        
        # Assuming edge_features are per unique pipe (as in your current code)
        # If your GNN expects edge features for bidirectional edges, this might need adjustment
        # For now, sticking to your current edge_features shape of (self.max_pipes, 4)

        edge_index_array = np.zeros((2, self.max_pipes * 2), dtype=np.int32) # For bidirectional graph edges

        # --- Map node names to a consistent integer index ---
        node_list = list(self.current_network.junctions()) # Consider if other node types should be here
        node_name_to_idx = {name: i for i, (name, _) in enumerate(node_list)}
        num_nodes = len(node_list)

        # --- Extract Node Features ---
        for i, (node_name, node) in enumerate(node_list):
            if i < self.max_nodes:
                node_features[i] = [
                    node.base_demand if hasattr(node, 'base_demand') and node.base_demand is not None else 0.0,
                    node.elevation if hasattr(node, 'elevation') else 0.0,
                    self.node_pressures.get(node_name, 0.0),
                    1.0  # is_junction (assuming only junctions are in node_list for now)
                ]
        
        # --- Extract Edge Features and Edge Index ---
        edge_idx_list_for_gnn = [] # For GNN's edge_index (bidirectional)
        actual_num_pipes = len(self.pipe_names)

        for i, pipe_name in enumerate(self.pipe_names):
            if i < self.max_pipes: # For edge_features array
                pipe = self.current_network.get_link(pipe_name)
                is_current_pipe = 1.0 if i == self.current_pipe_index else 0.0
                edge_features[i] = [pipe.diameter, pipe.length, pipe.roughness, is_current_pipe]
                
                start_node_idx = node_name_to_idx.get(pipe.start_node_name)
                end_node_idx = node_name_to_idx.get(pipe.end_node_name)

                if start_node_idx is not None and end_node_idx is not None:
                    edge_idx_list_for_gnn.append([start_node_idx, end_node_idx])
                    edge_idx_list_for_gnn.append([end_node_idx, start_node_idx])

        if edge_idx_list_for_gnn:
            edge_index_tensor_gnn = np.array(edge_idx_list_for_gnn, dtype=np.int32).T
            # Ensure slicing does not go out of bounds for edge_index_array
            cols_to_copy = min(edge_index_tensor_gnn.shape[1], edge_index_array.shape[1])
            edge_index_array[:, :cols_to_copy] = edge_index_tensor_gnn[:, :cols_to_copy]


        # --- Extract Global Features ---
        global_features = np.array([num_nodes, actual_num_pipes, self.current_pipe_index], dtype=np.float32)

        # --- Get the current action mask ---
        # This mask is crucial for MaskablePPO
        # current_action_mask = self.get_action_mask().astype(np.int8) # Ensure correct dtype

        """Action mask not supported for custom environment setup in stable baselines3"""

        return {
            "nodes": node_features,
            "edges": edge_features,
            "edge_index": edge_index_array,
            "globals": global_features,
            # "action_mask": current_action_mask
        }

    # Ensure your _get_zero_observation() method (if you have one) also includes a zeroed 'action_mask'
    def _get_zero_observation(self) -> Dict[str, np.ndarray]: # Example if you need this
        return {
            "nodes": np.zeros((self.max_nodes, 4), dtype=np.float32),
            "edges": np.zeros((self.max_pipes, 4), dtype=np.float32),
            "edge_index": np.zeros((2, self.max_pipes * 2), dtype=np.int32),
            "globals": np.zeros((3,), dtype=np.float32),
            # "action_mask": np.ones((self.action_space.n,), dtype=np.int8) # Default to all actions valid if obs is zeroed
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, scenario_name: Optional[str] = None) -> Tuple[Dict, Dict]:

        super().reset(seed=seed)
        
        # Add this logic to select the scenario
        if scenario_name and scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            print(f"INFO: Running specified scenario: {self.current_scenario}")
        else:
            self.current_scenario = self.np_random.choice(self.scenarios)
        
        self.network_states = self.load_network_states(self.current_scenario)
        
        self.current_time_step = 0
        self.current_pipe_index = 0
        
        self.actions_this_timestep = []
        self.chosen_diameters_from_previous_step = None
        self.chosen_roughness_from_previous_step = None
        self._load_network_for_timestep()
        
        obs = self.get_network_features()
        info = {'scenario': self.current_scenario, 'time_step': 0, 'pipe_index': 0}
        
        return obs, info

    def _load_network_for_timestep(self):
        """Helper function to load a network and set its initial state."""
        network_path = self.network_states[self.current_time_step]
        self.current_network = wntr.network.WaterNetworkModel(network_path) # Loads the next .inp file
        self.pipe_names = self.current_network.pipe_name_list
        
        # ADD THIS BLOCK to apply diameters from the previous step
        if self.chosen_diameters_from_previous_step:
            # print(f"INFO: Applying {len(self.chosen_diameters_from_previous_step)} chosen diameters from previous step to new network.")
            for pipe_name, diameter in self.chosen_diameters_from_previous_step.items():
                # Only update the pipe if it exists in the newly loaded network
                if pipe_name in self.current_network.pipe_name_list:
                    pipe = self.current_network.get_link(pipe_name)
                    pipe.diameter = diameter

        if self.chosen_roughness_from_previous_step:
            # print(f"INFO: Applying {len(self.chosen_roughness_from_previous_step)} chosen roughness values from previous step.")
            for pipe_name, roughness_val in self.chosen_roughness_from_previous_step.items():
                if pipe_name in self.current_network.pipe_name_list: #
                    pipe = self.current_network.get_link(pipe_name) #
                    pipe.roughness = roughness_val 

        if self.current_time_step > 0:
            print(f"INFO: Time step {self.current_time_step}. Decreasing H-W coeff for all pipes by 0.025.")
            for pipe_name in self.current_network.pipe_name_list: #
                pipe = self.current_network.get_link(pipe_name) #
                original_roughness = pipe.roughness
                pipe.roughness -= 0.025
                # print(f"  Pipe {pipe_name}: roughness {original_roughness:.4f} -> {pipe.roughness:.4f}")

        self.original_diameters_this_timestep = {p: self.current_network.get_link(p).diameter for p in self.pipe_names}

        try:
            wn_copy_pd = wntr.network.WaterNetworkModel(network_path)
            min_diameter = min(p['diameter'] for p in self.pipes.values())
            for p_name in wn_copy_pd.pipe_name_list:
                wn_copy_pd.get_link(p_name).diameter = min_diameter
            max_pd_results, _ = self.simulate_network(wn_copy_pd)
            if max_pd_results:
                max_pd_metrics = evaluate_network_performance(wn_copy_pd, max_pd_results)
                self.current_max_pd = max_pd_metrics.get('total_pressure_deficit', 0.0)
        except Exception as e:
            print(f"Warning: Failed to pre-calculate max_pd: {e}")
            self.current_max_pd = 0.0

        try:
            max_diameter = max(p['diameter'] for p in self.pipes.values())
            max_actions = [(pipe_id, max_diameter) for pipe_id in self.pipe_names]
            # Note: The energy_cost component will be based on the initial state, which is acceptable for a static ceiling.
            _, initial_metrics = self.simulate_network(self.current_network)
            initial_energy_cost = initial_metrics.get('total_pump_cost', 0.0) if initial_metrics else 0.0
            
            self.current_max_cost = compute_total_cost(list(self.current_network.pipes()), max_actions, self.labour_cost, initial_energy_cost, self.pipes, self.original_diameters_this_timestep)
        except Exception as e:
            print(f"Warning: Failed to pre-calculate max_cost: {e}")
            self.current_max_cost = 0.0
        
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
        action_is_valid_upgrade = False

        # Action 0 is "do nothing"
        if action == 0:
            action_is_valid_upgrade = True
        elif action > 0:
            new_diameter = self.pipe_diameter_options[action - 1]
            if new_diameter > old_diameter:
                self.current_network.get_link(pipe_name).diameter = new_diameter
                self.current_network.get_link(pipe_name).roughness = 150.0 # Reset H-W
                self.actions_this_timestep.append((pipe_name, new_diameter))
                action_is_valid_upgrade = True

        # # If the chosen action was not a valid upgrade, the episode should be penalized
        # if not action_is_valid_upgrade:
        #     # Penalize and end the episode immediately for taking an invalid action

        #     print(f"WARNING: Invalid action {action} on pipe {pipe_name} at index {self.current_pipe_index}. Original diameter: {old_diameter}, Action diameter: {self.pipe_diameter_options[action - 1] if action > 0 else 'do nothing'}")

        #     reward = -10.0 # A large penalty
        #     terminated = True
        #     obs = self.get_network_features()
        #     return obs, reward, terminated, truncated, {} # Return immediately

        self.current_pipe_index += 1

        # --- Logic for when all pipes have been decided for the current network state ---
        if self.current_pipe_index >= len(self.pipe_names):
            # First, simulate the network with the agent's chosen actions
            results, metrics = self.simulate_network(self.current_network)
            
            if results:

                reward_tuple = calculate_reward(
                    self.current_network, 
                    self.original_diameters_this_timestep, 
                    self.actions_this_timestep, 
                    self.pipes, 
                    metrics, 
                    self.labour_cost, 
                    False, 
                    False, 
                    [],
                    max_pd=self.current_max_pd,
                    max_cost=self.current_max_cost # Pass max_cost
                )

                # reward_tuple = reward_just_pd(
                #     self.current_network,
                #     self.original_diameters_this_timestep,
                #     self.actions_this_timestep,
                #     self.pipes,
                #     metrics, # performance_metrics
                #     self.labour_cost,
                #     False, # downgraded_pipes (assuming agent won't/can't downgrade)
                #     False, # disconnections (assuming not modeled or handled separately)
                #     [],    # actions_causing_disconnections
                #     max_pd=self.current_max_pd, # May not be used by reward_just_pd
                #     max_cost=self.current_max_cost # May not be used by reward_just_pd
                # )

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
                reward = 0.0 # Keep reward in normal value range
                info = {} # Return empty info on failure
            
            # --- Reset for the next major timestep in the scenario ---

            self.chosen_diameters_from_previous_step = {
                p_name: self.current_network.get_link(p_name).diameter 
                for p_name in self.pipe_names
            }

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