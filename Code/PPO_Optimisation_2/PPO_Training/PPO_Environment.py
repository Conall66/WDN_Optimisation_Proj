
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

    def _load_network_states(self, scenario: str) -> Dict[int, str]:
        """Load all network state files for a given scenario."""
        scenario_path = os.path.join(self.networks_folder, scenario)
        inp_files = [f for f in os.listdir(scenario_path) if f.endswith('.inp')]
        inp_files.sort()  # Ensure consistent ordering
        
        network_states = {}
        for i, inp_file in enumerate(inp_files):
            network_states[i] = os.path.join(scenario_path, inp_file)
        
        return network_states
    
    def _get_network_features(self, network: wntr.network.WaterNetworkModel) -> np.ndarray:
        """Extract features from the current network state."""
        features = []
        
        # Current time step normalized
        features.append(self.current_time_step / self.time_steps)
        
        # Current pipe index
        features.append(self.current_pipe_index)
        
        # Current pipe properties
        current_pipe = network.get_link(self.pipe_names[self.current_pipe_index])
        features.extend([
            current_pipe.diameter,
            current_pipe.length,
            current_pipe.roughness
        ])
        
        # Network statistics
        pipe_diameters = [network.get_link(pipe).diameter for pipe in self.pipe_names]
        features.extend([
            np.mean(pipe_diameters),
            np.std(pipe_diameters),
            np.min(pipe_diameters),
            np.max(pipe_diameters)
        ])
        
        # Demand information
        total_demand = sum([node.demand_timeseries_list[0].base_value 
                           for node in network.junctions()])
        features.append(total_demand)
        
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
        
    def reward(self, network, performance_metrics) -> float:
        """Calculate the reward based on the performance metrics."""
        return calculate_reward(performance_metrics)
        
    

