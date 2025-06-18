"""
In this file, we take the .inp files from the Networks2 folder and create a gym environment for each network.

MODIFIED to:
- Calculate and store baseline performance metrics at the start of each episode for reward normalization.
- Apply a hard penalty (reward=0) if the budget becomes negative.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wntr
import os
from typing import Dict, List, Tuple, Optional

# Import reward functions and hydraulic model helpers
from Hydraulic_2 import run_epanet_simulation, evaluate_network_performance
from Reward2 import calculate_reward, compute_total_cost, _reward_custom_normalized

class WNTRGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            pipes_config: Dict,
            scenarios: List[str],
            network_config: Dict,
            budget_config: Dict,
            reward_config: Dict,
            networks_folder: str = 'Networks2'
            ):
        """
        Initializes the WNTR Gym Environment.
        """
        super(WNTRGymEnv, self).__init__()

        # === Unpack Configurations ===
        self.pipes_config = pipes_config
        self.pipe_diameter_options = sorted([p['diameter'] for p in pipes_config.values()])
        self.scenarios = scenarios
        self.networks_folder = networks_folder
        self.labour_cost = budget_config.get('labour_cost_per_meter', 100.0)

        # Network and Observation Space Config
        self.max_nodes = network_config.get('max_nodes', 150)
        self.max_pipes = network_config.get('max_pipes', 200)

        # Budget Config
        self.start_of_episode_budget = budget_config.get('start_of_episode_budget', 200000.0)
        self.initial_budget_per_step = budget_config.get('initial_budget_per_step', 100000.0)
        # The 'ongoing_debt_penalty_factor' is no longer used with the new hard penalty.
        # self.ongoing_debt_penalty_factor = budget_config.get('ongoing_debt_penalty_factor', 0.0001)
        self.max_debt = budget_config.get('max_debt', 2000000.0)

        # Reward Config
        self.reward_config = reward_config
        # Ensure the mode is set correctly for the new reward function
        self.reward_mode = reward_config.get('mode', 'custom_normalized')
        
        # Internal state variables
        self.current_scenario = None
        self.num_time_steps = 0
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.network_states = {}
        self.current_network = None
        self.pipe_names = []
        self.node_names = []
        self.node_pressures = {}
        self.chosen_diameters_from_previous_step = {}
        self.chosen_roughness_from_previous_step = {}
        self.cumulative_budget = 0.0

        self.actions_this_timestep = []
        
        # NEW: Baseline metrics for reward normalization, set at the start of each episode
        self.baseline_pressure_deficit = 0.0
        self.baseline_demand_satisfaction = 0.0

        # === Action Space: N options + 1 for 'do nothing' ===
        self.action_space = spaces.Discrete(len(self.pipe_diameter_options) + 1)

        # === Observation Space (Dictionary format for GNN) ===
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, 4), dtype=np.float32),
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_pipes, 4), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.max_nodes, shape=(2, self.max_pipes * 2), dtype=np.int32),
            "globals": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        # Select a scenario for the episode
        scenario_name = options.get('scenario_name') if options else None
        if scenario_name and scenario_name in self.scenarios:
            self.current_scenario = scenario_name
        else:
            self.current_scenario = self.np_random.choice(self.scenarios)

        self.network_states = self._load_network_states(self.current_scenario)
        self.num_time_steps = len(self.network_states)

        # Initialize budget and episode state
        self.cumulative_budget = self.start_of_episode_budget
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.chosen_diameters_from_previous_step = {}
        self.chosen_roughness_from_previous_step = {}

        self.actions_this_timestep = []

        # Load the initial network state for t=0
        self._load_network_for_timestep()
        self.cumulative_budget += self.initial_budget_per_step # Add budget for the first period
        
        # NEW: Calculate and store baseline metrics for the entire episode
        initial_metrics = self._get_initial_metrics()
        self.baseline_pressure_deficit = initial_metrics.get('total_pressure_deficit', 0.0)
        self.baseline_demand_satisfaction = initial_metrics.get('demand_satisfaction_ratio', 0.0)

        obs = self._get_network_features()
        info = {
            'scenario': self.current_scenario,
            'time_step': self.current_time_step,
            'initial_metrics': initial_metrics
        }
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        terminated = False
        truncated = False
        info = {}
        
        # Apply action to the current pipe
        pipe_name = self.pipe_names[self.current_pipe_index]
        old_diameter = self.current_network.get_link(pipe_name).diameter
        action_tuple = None
        
        if action > 0:
            new_diameter = self.pipe_diameter_options[action - 1]
            if new_diameter > old_diameter:
                self.current_network.get_link(pipe_name).diameter = new_diameter
                self.current_network.get_link(pipe_name).roughness = 150.0 
                action_tuple = (pipe_name, new_diameter)
                self.actions_this_timestep.append(action_tuple)

        # Run simulation and get metrics
        results, metrics, sim_success = self._simulate_network(self.current_network)

        actions_for_cost = [action_tuple] if action_tuple else []
        cost_of_intervention = compute_total_cost(
            actions=actions_for_cost,
            pipes_config=self.pipes_config,
            wn=self.current_network,
            energy_cost=metrics.get('total_pump_cost', 0) if sim_success else float('inf'),
            labour_cost_per_meter=self.labour_cost
        )
        
        # Prepare parameters for the NEW reward function, including baselines
        reward_params = {
            'metrics': metrics,
            'cost_of_intervention': cost_of_intervention,
            'baseline_pressure_deficit': self.baseline_pressure_deficit,
            'baseline_demand_satisfaction': self.baseline_demand_satisfaction,
            'current_budget': self.cumulative_budget,  # Pass current budget before deduction
            **self.reward_config 
        }

        if sim_success:
            # Get the reward (now including budget penalty)
            # reward, reward_components = calculate_reward(mode=self.reward_mode, params=reward_params)
            reward, reward_components = _reward_custom_normalized(params=reward_params)
            
            # Update budget (but don't apply penalty here anymore)
            budget_before_spending = self.cumulative_budget
            self.cumulative_budget -= cost_of_intervention
            
            # Truncate if debt becomes excessive
            if self.cumulative_budget < -self.max_debt:
                # truncated = True
                reward = 0.0  # Also ensure reward is zero on final truncation step

            info = {
                'step_reward': reward, 'base_reward': reward,
                'cost_of_intervention': cost_of_intervention,
                'pressure_deficit': metrics.get('total_pressure_deficit', -1),
                'demand_satisfaction': metrics.get('demand_satisfaction_ratio', -1),
                'pipe_changes': len(actions_for_cost),
                'budget_before_step': budget_before_spending,
                'cumulative_budget': self.cumulative_budget,
                'simulation_success': True,
                **reward_components
            }
        else: # Simulation failed
            reward = 0.0 # Reward must be 0 for failure
            truncated = True
            info = {'error': 'Simulation failed', 'simulation_success': False, 'step_reward': reward}

        # Advance state to the next pipe or next major timestep
        if not terminated and not truncated:
            self.current_pipe_index += 1

            if self.current_pipe_index >= len(self.pipe_names):
                self.chosen_diameters_from_previous_step = {p: self.current_network.get_link(p).diameter for p in self.pipe_names}
                self.chosen_roughness_from_previous_step = {p: self.current_network.get_link(p).roughness for p in self.pipe_names}
                self.current_time_step += 1

                self.actions_this_timestep = []  # Reset actions for the next timestep
                
                if self.current_time_step >= self.num_time_steps:
                    terminated = True
                else:
                    self.current_pipe_index = 0
                    self.cumulative_budget += self.initial_budget_per_step
                    self._load_network_for_timestep()

        print(f"Step {self.current_time_step}, Pipe {self.current_pipe_index}, Action: {action}, Reward: {reward}, Budget: {self.cumulative_budget:.2f}")
        print(f"Reward Components: {reward_components}")
        print("=" * 50)

        obs = self._get_network_features()
        return obs, reward, terminated, truncated, info

    def _get_network_features(self) -> Dict[str, np.ndarray]:
        # This function does not need changes
        node_features = np.zeros((self.max_nodes, 4), dtype=np.float32)
        edge_features = np.zeros((self.max_pipes, 4), dtype=np.float32)
        edge_index_array = np.zeros((2, self.max_pipes * 2), dtype=np.int32)
        raw_node_list = list(self.current_network.nodes())
        all_node_names = [n[0] if isinstance(n, tuple) else n for n in raw_node_list]
        node_name_to_idx = {name: i for i, name in enumerate(all_node_names)}
        num_nodes = len(all_node_names)
        for i, node_name in enumerate(all_node_names):
            if i >= self.max_nodes: break
            node = self.current_network.get_node(node_name)
            is_junction = 1.0 if node.node_type == 'Junction' else 0.0
            demand = node.base_demand if hasattr(node, 'base_demand') and node.base_demand is not None else 0.0
            node_features[i] = [demand, node.elevation if hasattr(node, 'elevation') else 0.0, self.node_pressures.get(node_name, 0.0), is_junction]
        edge_idx_list_for_gnn = []
        actual_num_pipes = len(self.pipe_names)
        for i, pipe_name in enumerate(self.pipe_names):
            if i >= self.max_pipes: break
            pipe = self.current_network.get_link(pipe_name)
            edge_features[i] = [pipe.diameter, pipe.length, pipe.roughness, 1.0 if i == self.current_pipe_index else 0.0]
            start_node_idx = node_name_to_idx.get(pipe.start_node_name)
            end_node_idx = node_name_to_idx.get(pipe.end_node_name)
            if start_node_idx is not None and end_node_idx is not None:
                edge_idx_list_for_gnn.extend([[start_node_idx, end_node_idx], [end_node_idx, start_node_idx]])
        if edge_idx_list_for_gnn:
            edge_index_tensor_gnn = np.array(edge_idx_list_for_gnn, dtype=np.int32).T
            cols_to_copy = min(edge_index_tensor_gnn.shape[1], edge_index_array.shape[1])
            edge_index_array[:, :cols_to_copy] = edge_index_tensor_gnn[:, :cols_to_copy]
        normalizing_budget = self.start_of_episode_budget or 1.0
        normalized_budget = min(self.cumulative_budget / normalizing_budget, 5.0)
        global_features = np.array([num_nodes, actual_num_pipes, self.current_pipe_index, normalized_budget], dtype=np.float32)
        return {"nodes": node_features, "edges": edge_features, "edge_index": edge_index_array, "globals": global_features}

    def _load_network_for_timestep(self):
        # This function does not need changes
        network_path = self.network_states[self.current_time_step]
        self.current_network = wntr.network.WaterNetworkModel(network_path)
        self.pipe_names = self.current_network.pipe_name_list
        self.node_names = self.current_network.node_name_list
        
        # Apply saved diameters to existing pipes
        for pipe_name, diameter in self.chosen_diameters_from_previous_step.items():
            if pipe_name in self.current_network.pipe_name_list:
                self.current_network.get_link(pipe_name).diameter = diameter
        
        # NEW: Initialize any new pipes with appropriate default diameters
        for pipe_name in self.pipe_names:
            if pipe_name not in self.chosen_diameters_from_previous_step:
                # Use either the existing diameter or the smallest option from pipes_config
                current_diameter = self.current_network.get_link(pipe_name).diameter
                if current_diameter <= 0.01:  # If it's unreasonably small
                    smallest_diameter = min(p['diameter'] for p in self.pipes_config.values())
                    self.current_network.get_link(pipe_name).diameter = smallest_diameter
                    self.current_network.get_link(pipe_name).roughness = 130.0  # Default roughness
        
        # After setting all diameters, save current state
        self.chosen_diameters_from_previous_step = {p: self.current_network.get_link(p).diameter for p in self.pipe_names}
        self.chosen_roughness_from_previous_step = {p: self.current_network.get_link(p).roughness for p in self.pipe_names}

        # DEBUG HERE: print the diameters of all pipes in the current network
        # print(f"Loaded network for timestep {self.current_time_step} with {len(self.pipe_names)} pipes.")
        # print("Pipe diameters:")
        # for pipe_name in self.pipe_names:
        #     pipe = self.current_network.get_link(pipe_name)
        #     print(f"  {pipe_name}: diameter={pipe.diameter:.4f}m, roughness={pipe.roughness:.2f}")
    
        results, _, _ = self._simulate_network(self.current_network)
        if results:
            # print(f"Simulation successful for timestep {self.current_time_step}.")
            pressures = results.node['pressure']
            self.node_pressures = {node_name: pressures.loc[:, node_name].mean() for node_name in self.node_names if node_name in pressures.columns}
        else:
            # print(f"Simulation failed for timestep {self.current_time_step}. Setting all node pressures to 0.")
            self.node_pressures = {node_name: 0.0 for node_name in self.node_names}

    def _get_initial_metrics(self):
        # This function does not need changes
        _, metrics, _ = self._simulate_network(self.current_network)
        return metrics

    def _simulate_network(self, network: wntr.network.WaterNetworkModel):
        # This function does not need changes
        try:
            results = run_epanet_simulation(network)
            if results is None or results.node['pressure'].isnull().values.any():
                return None, {}, False
            metrics = evaluate_network_performance(network, results)
            # print(f"Simulation completed for scenario '{self.current_scenario}' at time step {self.current_time_step}.")
            return results, metrics, True
        except Exception:
            return None, {}, False
            
    def _load_network_states(self, scenario: str) -> Dict[int, str]:
        # This function does not need changes
        scenario_path = os.path.join(self.networks_folder, scenario)
        if not os.path.isdir(scenario_path):
            raise FileNotFoundError(f"Scenario folder not found: {scenario_path}")
        inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return {i: os.path.join(scenario_path, f) for i, f in enumerate(inp_files)}

    def close(self):
        pass

if __name__ == "__main__":

    # Use the same configuration dictionaries from your training script
    PIPES_CONFIG = {'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
                    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32}, 
                    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
                    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47}, 
                    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
                    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}}
    
    NETWORK_CONFIG = {'max_nodes': 150, 'max_pipes': 200}
    BUDGET_CONFIG = {"initial_budget_per_step": 200000.0, "start_of_episode_budget": 5000000.0}
    REWARD_CONFIG = {'mode': 'custom_normalized'}
    SCENARIOS = ['hanoi_sprawling_3'] # Test with one scenario for simplicity

    print("Initializing WNTRGymEnv...")
    try:
        env = WNTRGymEnv(
            pipes_config=PIPES_CONFIG,
            scenarios=SCENARIOS,
            network_config=NETWORK_CONFIG,
            budget_config=BUDGET_CONFIG,
            reward_config=REWARD_CONFIG
        )
        print("Environment initialized successfully.")

        # Test the reset method
        obs, info = env.reset()
        print("\n--- Reset Test ---")
        print("Reset successful. Initial observation received.")
        print(f"Initial budget: ${env.cumulative_budget:.2f}")
        
        # CRITICAL: Check the structure and shape of the observation
        print("\n--- Observation Structure Check ---")
        for key, value in obs.items():
            print(f"Key: '{key}', Type: {type(value)}, Shape: {value.shape}, Dtype: {value.dtype}")
        
        expected_shapes = {
            "nodes": (NETWORK_CONFIG['max_nodes'], 4),
            "edges": (NETWORK_CONFIG['max_pipes'], 4),
            "edge_index": (2, NETWORK_CONFIG['max_pipes'] * 2),
            "globals": (4,)
        }
        
        is_ok = all(obs[k].shape == s for k, s in expected_shapes.items())
        print(f"\nObservation shapes match expected for GNN: {'✅' if is_ok else '❌'}")


        # Test the step method with a random action
        print("\n--- Step Test (taking 5 random steps) ---")
        for i in range(100):
            action = env.action_space.sample()
            pipe_name = env.pipe_names[env.current_pipe_index] if env.current_pipe_index < len(env.pipe_names) else "Unknown"
            old_diameter = env.current_network.get_link(pipe_name).diameter if env.current_pipe_index < len(env.pipe_names) else 0
            
            print(f"\nStep {i+1}:")
            print(f"Current budget: ${env.cumulative_budget:.2f}")
            print(f"Pipe: {pipe_name}, Current diameter: {old_diameter:.4f}m")
            
            if action == 0:
                print(f"Action: {action} (Do nothing)")
            else:
                new_diameter = env.pipe_diameter_options[action - 1]
                print(f"Action: {action} (Change to diameter {new_diameter:.4f}m)")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
             # Display key information about the step
            print(f"Cost of action: ${info.get('cost_of_intervention', 'N/A'):.2f}")
            print(f"Reward: {reward:.4f}")
            
            # Display the normalized weighted reward components
            if 'normalized_pressure_deficit' in info:
                print(f"Normalized pressure deficit: {info['normalized_pressure_deficit']:.4f}")
            if 'normalized_demand_satisfaction' in info:
                print(f"Normalized demand satisfaction: {info['normalized_demand_satisfaction']:.4f}")
            if 'normalized_cost' in info:
                print(f"Normalized cost: {info['normalized_cost']:.4f}")
                
            # Display the weighted components if available
            if 'weighted_pressure_component' in info:
                print(f"Weighted pressure component: {info['weighted_pressure_component']:.4f}")
            if 'weighted_demand_component' in info:
                print(f"Weighted demand component: {info['weighted_demand_component']:.4f}")
            if 'weighted_cost_component' in info:
                print(f"Weighted cost component: {info['weighted_cost_component']:.4f}")
            
            # Display the raw metrics for comparison
            if 'pressure_deficit' in info:
                print(f"Raw pressure deficit: {info['pressure_deficit']:.4f}")
            if 'demand_satisfaction' in info:
                print(f"Raw demand satisfaction: {info['demand_satisfaction']:.4f}")
            
            print(f"Simulation success: {info.get('simulation_success', False)}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print(f"Episode ended after step {i+1}")
                break
        
        env.close()
        print("\n✅ Environment test completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during environment test: {e}")