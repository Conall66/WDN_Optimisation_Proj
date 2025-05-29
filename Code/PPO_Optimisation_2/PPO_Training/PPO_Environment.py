
"""

In this file, we take the .inp files from the Modified_nets folder and create a gym environment for each network.

"""

import gym
from gym import spaces
import numpy as np
import wntr
import os
import random
from typing import Dict, List, Tuple, Optional
import pandas as pd

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import calculate_reward

class WNTRGymEnv(gym.Env):
    def __init__(
            self,
            pipes, # Dictionary with pipe {Pipe_ID: Pipe Diameter, Unit Cost}
            roughnesses, # Hazen-Wiliams roughness values
            scenarios,
            networks_folder='Modified_nets',
            pressure_threshold = 0,
            max_epsiodes = 1000):
        
        super(WNTRGymEnv, self).__init__()

        self.pipes = pipes
        self.roughnesses = roughnesses
        self.scenarios = scenarios
        self.networks_folder = networks_folder
        self.pressure_threshold = pressure_threshold
        self.max_epsiodes = max_epsiodes
        self.labour_cost = 100 # (£/m) Installed of pipe
        
        self.current_scenario = None
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.network_states = {}
        self.current_network = None
        self.pipe_names = []
        self.episode_count = 0
        
        self.start_year = 2025
        self.end_year = 2050
        self.time_steps = 50 # One for each 6 month window

        self.simulation_duration = 24 * 3600
        self.simulation_time_step = 3600

        # Define action space: 0 = no change, 1-N = change to diameter option N
        self.action_space = spaces.Discrete(len(self.pipe_diameter_options) + 1)
        
        # Define observation space (will be set after first reset)
        self.observation_space = None

    def load_network_states(self, scenario: str) -> Dict[int, str]:
        """Load all network state files for a given scenario."""
        scenario_path = os.path.join(self.networks_folder, scenario)
        inp_files = [f for f in os.listdir(scenario_path) if f.endswith('.inp')]
        inp_files.sort()  # Ensure consistent ordering
        
        network_states = {}
        for i, inp_file in enumerate(inp_files):
            network_states[i] = os.path.join(scenario_path, inp_file)
        
        return network_states
    
    def get_network_features(self, network: wntr.network.WaterNetworkModel) -> np.ndarray:
        """Extract features from the current network state."""
        features = []
        
        # Local features include the pipe diameters, node demands and nodal pressures
        diameters = []
        for pipe in network.pipes():
            pipe_id = pipe[0]
            pipe_data = pipe[1]
            diameter = self.pipes.get(pipe_id, 0)
            diameters.append(diameter)

        demands = []
        for node, node_data in network.junctions():
            demand = node_data.demand
            demands.append(demand)

        pressures = []
        for node, node_data in network.junctions():
            pressure = node_data.pressure
            pressures.append(pressure)

        # Global features of current pipe ID
        current_pipe_id = self.pipe_names[self.current_pipe_index] if self.pipe_names else None

        # Add all features to the list
        features.extend(diameters)
        features.extend(demands)
        features.extend(pressures)
        features.append(current_pipe_id if current_pipe_id else 0)    

        return np.array(features, dtype=np.float32)
    
    def simulate_network(self, network: wntr.network.WaterNetworkModel):

        """Run the EPANET simulation for the current network."""
        try:
            results = run_epanet_simulation(network)
            performance_metrics = evaluate_network_performance(network, results)
            return results, performance_metrics
        except Exception as e:
            print(f"Simulation failed: {e}")
            return None
        
    def reward(self, action, performance_metrics) -> float:
        """Calculate the reward based on the performance metrics."""
        
        return calculate_reward(
            self.current_network,
            action,
            self.pipes,
            performance_metrics,
            self.labour_cost
        )
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        # Select random scenario
        self.current_scenario = random.choice(self.scenarios)
        self.network_states = self.load_network_states(self.current_scenario)
        
        # Reset episode state
        self.current_time_step = 0
        self.current_pipe_index = 0
        
        # Load initial network
        initial_network_path = self.network_states[0]
        self.current_network = wntr.network.WaterNetworkModel(initial_network_path)
        self.pipe_names = list(self.current_network.pipe_name_list)
        
        # Set observation space if not already set
        if self.observation_space is None:
            sample_obs = self.get_network_features(self.current_network)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
            )
        
        self.episode_count += 1
        
        return self.get_network_features(self.current_network)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:

        """Action = [
        0: No change,
        1: Change pipe diameter to the smallest diameter pipe
        2: Change pipe diameter to the second smallest diameter pipe,
        ...
        N: Change pipe diameter to the largest diameter pipe
        ]"""

        if action > 0:  # Change pipe diameter
            new_diameter = self.pipe_diameter_options[action - 1] # 0 Indexing in python
            pipe = self.current_network.get_link(self.pipe_names[self.current_pipe_index])
            old_diameter = pipe.diameter
            pipe.diameter = new_diameter
            
            # Calculate cost of change
            # cost_change = (self.pipe_cost_per_meter[new_diameter] - 
            #               self.pipe_cost_per_meter.get(old_diameter, old_diameter * 1000)) * pipe.length
            """ Cost of change encapsulated in reward function"""

        else:
            cost_change = 0
        
        # Move to next pipe
        self.current_pipe_index += 1
        
        # Check if all pipes processed for current time step
        if self.current_pipe_index >= len(self.pipe_names):
            # Evaluate network performance
            serviced_demand, pressure_deficit, total_cost = self.simulate_network(self.current_network)
            
            # Calculate reward
            reward = self.reward(serviced_demand, pressure_deficit, cost_change)
            
            # Move to next time step
            self.current_time_step += 1
            self.current_pipe_index = 0
            
            # Load next network state if available
            if self.current_time_step < len(self.network_states):
                next_network_path = self.network_states[self.current_time_step]
                # Update network with evolved topology but keep optimized diameters
                temp_network = wntr.network.WaterNetworkModel(next_network_path)
                
                # Transfer optimized diameters to new network topology
                for pipe_name in self.pipe_names:
                    if pipe_name in temp_network.pipe_name_list:
                        old_pipe = self.current_network.get_link(pipe_name)
                        new_pipe = temp_network.get_link(pipe_name)
                        new_pipe.diameter = old_pipe.diameter
                
                self.current_network = temp_network
                self.pipe_names = list(self.current_network.pipe_name_list)
        else:
            reward = 0  # No reward until time step is complete
        
        # Check if episode is done
        done = (self.current_time_step >= len(self.network_states) or 
                self.episode_count >= self.max_episodes)
        
        # Get next observation
        obs = self._get_network_features(self.current_network)
        
        # Info dictionary
        info = {
            'scenario': self.current_scenario,
            'time_step': self.current_time_step,
            'pipe_index': self.current_pipe_index,
            'total_pipes': len(self.pipe_names),
            'episode': self.episode_count
        }
        
        return obs, reward, done, info
    

