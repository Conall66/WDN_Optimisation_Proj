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
# import random
from typing import Dict, List, Tuple, Optional
# import networkx as nx
import shutil
import multiprocessing

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import calculate_reward, reward_just_pd, compute_total_cost, reward_minimise_pd, reward_pd_and_cost, reward_full_objective

multiprocessing.set_start_method('spawn', force=True)

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
            current_max_pd = 5000000.0, #HARD CODED ESTIMATES FOR TESTING
            current_max_cost = 2000000.0, # HARD CODED ESTIMATES FOR TESTING
            reward_mode: str = 'full_objective',  # Options: 'minimise_pd', 'pd_and_cost', 'full_objective'
            initial_budget_per_step: float = 100000.0,  # Example: Budget added each major step
            start_of_episode_budget: float = 200000.0, # Example: Lump sum at episode start
            # budget_exceeded_penalty_type: str = "set_to_zero", # Options: "set_to_zero", "multiplicative"
            # budget_penalty_factor: float = 0.5 # Used if penalty_type is "multiplicative"
            ongoing_debt_penalty_factor: float = 0.0001, # Penalty per unit of debt, per step
            max_debt: float = 2000000.0 # Terminate episode if debt exceeds this value
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
        self.reward_mode = reward_mode
        
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

        self.initial_budget_per_step = initial_budget_per_step
        self.start_of_episode_budget = start_of_episode_budget
        # self.budget_exceeded_penalty_type = budget_exceeded_penalty_type
        # self.budget_penalty_factor = budget_penalty_factor
        self.ongoing_debt_penalty_factor = ongoing_debt_penalty_factor
        self.max_debt = max_debt
        # This will now be the single source of truth for the episode's budget
        self.cumulative_budget = 0.0
        self.current_step_budget_available = 0.0 # Budget for the current major step decisions
        # self.spent_on_interventions_this_step = 0.0 # This will be cost_of_intervention

        # pid = os.getpid()
        # self.temp_dir = f"wntr_temp_{pid}"
        # os.makedirs(self.temp_dir, exist_ok=True)
        
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
            "globals": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

            # ACTION MASK REMOVED
        })

    def _initialize_episode_budget(self):
        """Initializes or resets the budget at the start of a new episode."""
        self.current_step_budget_available = self.start_of_episode_budget + self.initial_budget_per_step
        print(f"  Episode Start: Initial budget for Step 0 allocated: {self.current_step_budget_available:.2f}")

    def _apply_budget_penalty_and_update(self, cost_of_intervention: float, base_reward: float) -> Tuple[float, bool, float, float]:
        """
        Checks spending against budget, applies penalty to reward if exceeded,
        and calculates budget for the next step.

        Returns:
            Tuple (penalised_reward, budget_exceeded_flag, budget_before_spending, carry_over_for_next_step)
        """
        budget_available_for_this_step = self.current_step_budget_available
        penalised_reward = base_reward
        budget_exceeded = False

        if cost_of_intervention > budget_available_for_this_step:
            budget_exceeded = True
            if self.budget_exceeded_penalty_type == "set_to_zero":
                penalised_reward = 0.0
            elif self.budget_exceeded_penalty_type == "multiplicative":
                penalised_reward *= self.budget_penalty_factor
            
            print(f"BUDGET EXCEEDED: Spent {cost_of_intervention:.2f} > Budget {budget_available_for_this_step:.2f}. Base Reward {base_reward:.2f} -> penalised Reward {penalised_reward:.2f}")
        
        unspent_budget = budget_available_for_this_step - cost_of_intervention
        carry_over_budget = max(0.0, unspent_budget) # No negative carry-over (debt)

        return penalised_reward, budget_exceeded, budget_available_for_this_step, carry_over_budget


    def _calculate_episode_normalization_constants(self):
        """
        Calculates max_pd and max_cost for the initial state of the current episode's scenario.
        This should be called once in reset() after the scenario and its Step_0.inp are known.
        """
        if not self.network_states:
            print("Warning: Network states not loaded, cannot calculate episode normalization constants.")
            self.current_max_pd = 100000.0  # High fallback
            self.current_max_cost = 100000.0 # High fallback
            return

        initial_network_path_for_episode = self.network_states[0] # Based on Step_0.inp

        # Calculate episode_max_pd
        try:
            # Load a fresh copy for this calculation
            wn_for_pd_calc = wntr.network.WaterNetworkModel(initial_network_path_for_episode)
            min_diameter = min(p_data['diameter'] for p_data in self.pipes.values())
            for p_name in wn_for_pd_calc.pipe_name_list:
                pipe = wn_for_pd_calc.get_link(p_name)
                pipe.diameter = min_diameter
            
            # Use self.simulate_network as it handles errors and temp dirs
            max_pd_results, _, _ = self.simulate_network(wn_for_pd_calc)
            if max_pd_results:
                max_pd_metrics = evaluate_network_performance(wn_for_pd_calc, max_pd_results) #
                self.current_max_pd = max_pd_metrics.get('total_pressure_deficit', 1.0) 
                if self.current_max_pd <= 0: self.current_max_pd = 1.0 # Avoid DivByZero or non-positive
            else:
                print(f"Warning: Simulation for max_pd failed for {self.current_scenario} Step 0. Using fallback.")
                self.current_max_pd = 100000.0 # Fallback large value
        except Exception as e:
            print(f"ERROR during episode_max_pd calculation for {self.current_scenario} Step 0: {e}")
            self.current_max_pd = 100000.0 

        # Calculate episode_max_cost
        try:
            # Load a fresh copy for this calculation
            wn_initial_for_episode = wntr.network.WaterNetworkModel(initial_network_path_for_episode)
            max_diameter_opt = max(p_data['diameter'] for p_data in self.pipes.values())
            
            temp_orig_diams = {p_name: wn_initial_for_episode.get_link(p_name).diameter 
                               for p_name in wn_initial_for_episode.pipe_name_list}
            
            max_actions_list = []
            for p_name_mc in wn_initial_for_episode.pipe_name_list:
                # Cost is for upgrading to max_diameter_opt regardless of current state for max_cost scenario
                max_actions_list.append((p_name_mc, max_diameter_opt))

            initial_results, initial_metrics, sim_success = self.simulate_network(wn_initial_for_episode)
            base_energy_cost = initial_metrics.get('total_pump_cost', 0.0) if initial_metrics else 0.0 #
            
            self.current_max_cost = compute_total_cost(
                list(wn_initial_for_episode.pipes()), # Use list() for pipe iterator
                max_actions_list,
                self.labour_cost,
                base_energy_cost, 
                self.pipes,
                temp_orig_diams
            ) #
            if self.current_max_cost <= 0: self.current_max_cost = 10000.0 # Avoid DivByZero or non-positive
        except Exception as e:
            print(f"ERROR during episode_max_cost calculation for {self.current_scenario} Step 0: {e}")
            self.current_max_cost = 100000.0

        print(f"  Episode Norm Constants for '{self.current_scenario}': max_pd={self.current_max_pd:.2f}, max_cost={self.current_max_cost:.2f}")


    def load_network_states(self, scenario: str) -> Dict[int, str]:
        scenario_path = os.path.join(self.networks_folder, scenario)
        # inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])

        # sort inp_files by the step number in the filename
        inp_files = [f for f in os.listdir(scenario_path) if f.endswith('.inp')]
        inp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Assuming filenames like 'network_0.inp', 'network_1.inp', etc.

        # print(f"INFO: Found {len(inp_files)} .inp files in scenario '{scenario}': {inp_files}")

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

        normalizing_denominator_for_budget = self.start_of_episode_budget + self.initial_budget_per_step
        if normalizing_denominator_for_budget == 0: normalizing_denominator_for_budget = 1.0 # Avoid division by zero

        normalized_budget = self.current_step_budget_available / normalizing_denominator_for_budget
        # Clip to a reasonable range if desired, e.g., max 2.0 or 3.0 if budget can accumulate.
        normalized_budget = min(normalized_budget, 5.0) # Example: Cap at 5x initial step budget

        # --- Extract Global Features ---
        global_features = np.array([
            num_nodes, 
            actual_num_pipes, 
            self.current_pipe_index,
            normalized_budget # MODIFIED: Add normalized budget
            ], dtype=np.float32) #

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
            "globals": np.zeros((4,), dtype=np.float32),
            # "action_mask": np.ones((self.action_space.n,), dtype=np.int8) # Default to all actions valid if obs is zeroed
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, scenario_name: Optional[str] = None) -> Tuple[Dict, Dict]:

        super().reset(seed=seed)

        # self._initialize_episode_budget() # Initialize budget at the start of the episode
        # self._calculate_episode_normalization_constants() # Sets self.episode_max_pd, self.episode_max_cost #
        
        # Add this logic to select the scenario
        if scenario_name and scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            # print(f"INFO: Running specified scenario: {self.current_scenario}")
        else:
            self.current_scenario = self.np_random.choice(self.scenarios)
        
        self.network_states = self.load_network_states(self.current_scenario)

        # self._initialize_episode_budget() # Initialize budget at the start of the episode

        self.cumulative_budget = self.start_of_episode_budget + self.initial_budget_per_step

        self._calculate_episode_normalization_constants() # Sets self.episode_max_pd, self.episode_max_cost #
        
        self.current_time_step = 0
        self.current_pipe_index = 0
        
        self.actions_this_timestep = []
        self.chosen_diameters_from_previous_step = None
        self.chosen_roughness_from_previous_step = None
        self._load_network_for_timestep()
        
        obs = self.get_network_features()
        
        info = { #
            'scenario': self.current_scenario, 
            'time_step': self.current_time_step, 
            'pipe_index': self.current_pipe_index,
            'current_budget_for_step': self.current_step_budget_available # Initial budget for Step 0
        }
        
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
        
        self.node_pressures = {}
        results, _, _ = self.simulate_network(self.current_network)
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

        # # If the chosen action was not a valid upgrade, the episode should be penalised
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
            results, metrics, sim_success = self.simulate_network(self.current_network)

            print(f"INFO: Simulation for time step {self.current_time_step} completed. Success: {sim_success}")
            
            # Initialize reward and info
            reward = 0.0
            info = {}

            if sim_success:
                # 1. Calculate the base reward and cost of intervention for this step
                cost_of_intervention = compute_total_cost(
                    list(self.current_network.pipes()), 
                    self.actions_this_timestep, 
                    self.labour_cost, 
                    metrics.get('total_pump_cost', 0),
                    self.pipes,
                    self.original_diameters_this_timestep
                )

                reward_tuple = calculate_reward( 
                    self.current_network, self.original_diameters_this_timestep, self.actions_this_timestep,
                    self.pipes, metrics, self.labour_cost, False, False, [],
                    max_pd=self.current_max_pd, max_cost=self.current_max_cost
                )
                
                base_reward = reward_tuple[0]
                reward = base_reward # Start with the base reward

                # 2. Update the cumulative budget by subtracting the cost
                budget_before_spending = self.cumulative_budget
                self.cumulative_budget -= cost_of_intervention
                print(f"  BUDGET: Was {budget_before_spending:.2f}, Spent {cost_of_intervention:.2f}, Now {self.cumulative_budget:.2f}")

                # 3. Apply the ongoing debt penalty if the budget is negative
                debt_penalty = 0.0
                if self.cumulative_budget < 0:
                    # The penalty is proportional to the magnitude of the debt
                    debt_penalty = abs(self.cumulative_budget) * self.ongoing_debt_penalty_factor
                    reward -= debt_penalty
                    print(f"  PENALTY: In debt. Applying ongoing penalty of {debt_penalty:.2f}. Reward {base_reward:.2f} -> {reward:.2f}")

                # 4. Check for termination due to excessive debt
                if abs(self.cumulative_budget) > self.max_debt:
                    print(f"  TERMINATION: Debt of {-self.cumulative_budget:.2f} exceeds max_debt of {self.max_debt:.2f}.")
                    terminated = True
                    reward -= 10 # Apply a large final penalty for catastrophic debt
                
                # 5. Populate the info dictionary for logging
                info = { 
                    'reward': reward,
                    'base_reward': base_reward,
                    'cost_of_intervention': cost_of_intervention,
                    'pressure_deficit': reward_tuple[2],
                    'demand_satisfaction': reward_tuple[3], 
                    'pipe_changes': len(self.actions_this_timestep),
                    'budget_before_step': budget_before_spending,
                    'cumulative_budget': self.cumulative_budget,
                    'debt_penalty_applied': debt_penalty,
                    'simulation_success': sim_success,
                }

            else: # Simulation failed
                reward = -50.0  # Large penalty for simulation failure
                terminated = True 
                info = {
                    'error': 'Simulation failed',
                    'cumulative_budget': self.cumulative_budget,
                    'cost_of_intervention': 0,
                    'simulation_success': sim_success,
                }
            
            # --- State transition for the NEXT major step ---
            # Store the chosen diameters and roughness to apply to the next network state
            self.chosen_diameters_from_previous_step = {p_name: self.current_network.get_link(p_name).diameter for p_name in self.pipe_names}
            self.chosen_roughness_from_previous_step = {p_name: self.current_network.get_link(p_name).roughness for p_name in self.pipe_names}

            self.current_time_step += 1
            self.current_pipe_index = 0
            self.actions_this_timestep = []

            # Check for end of episode
            if self.current_time_step >= len(self.network_states):
                terminated = True

            if not terminated:
                # 6. Replenish the budget for the upcoming step and load the network
                self.cumulative_budget += self.initial_budget_per_step
                print(f"  BUDGET: Replenished by {self.initial_budget_per_step:.2f} for next step. New total: {self.cumulative_budget:.2f}\n")
                self._load_network_for_timestep()
        
        obs = self.get_network_features()
        return obs, reward, terminated, truncated, info

    def simulate_network(self, network: wntr.network.WaterNetworkModel):
        try:
            results = run_epanet_simulation(network)
            # Check for NaNs in the results right after simulation
            if results.node['pressure'].isnull().values.any():
                print(f"WARNING: NaN pressure values detected in scenario: {self.current_scenario}, time_step: {self.current_time_step}")
            metrics = evaluate_network_performance(network, results)
            sim_success = True
            return results, metrics, sim_success
        except Exception as e:
            # Log the error with context
            print(f"ERROR: Simulation failed for scenario: {self.current_scenario}, time_step: {self.current_time_step}. Error: {e}")
            sim_success = False
            return None, None, sim_success
            
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
        # shutil.rmtree(self.temp_dir)
        pass