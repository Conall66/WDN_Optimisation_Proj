
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
import shutil
import time
import copy
import networkx as nx

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Reward import calculate_reward

class WNTRGymEnv(gym.Env):
    def __init__(
            self,
            pipes, # Dictionary with pipe {Pipe_ID: Pipe Diameter, Unit Cost}
            scenarios,
            networks_folder='Modified_nets',
            pressure_threshold = 0,
            max_episodes = 1000000,
            ):
        
        super(WNTRGymEnv, self).__init__()

        self.pipes = pipes # {Pipe_ID: Pipe Diameter, Unit Cost}
        # self.pipe_IDs = list([pipe[0] for pipe in pipes.items()])
        self.pipe_IDs = list(pipes.keys())
        self.pipe_diameter_options = [pipe_data['diameter'] for pipe_data in pipes.values()]

        self.scenarios = scenarios
        self.networks_folder = networks_folder
        self.pressure_threshold = pressure_threshold
        self.max_episodes = max_episodes
        self.labour_cost = 100 # (£/m) Installed of pipe
        
        self.current_scenario = None # Gets rewritten in reset()
        self.current_time_step = 0
        self.current_pipe_index = 0
        self.network_states = {}
        self.current_network = None # gets rewritten in reset()
        self.pipe_names = []
        self.episode_count = 0
        self.node_pressures = {} # Dictionary allows for easy addition of new nodal pressures
        self.actions = []
        
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
        
        return network_states # Returns all 50 steps of a scenario
    
    def get_network_features(self, network: wntr.network.WaterNetworkModel) -> np.ndarray:
        """Extract features from the current network state."""
        features = []
        
        # Local features include the pipe diameters, node demands and nodal pressures
        diameters = []
        for pipe, pipe_data in network.pipes():
            diameter = pipe_data.diameter
            diameters.append(diameter)
        
        # print(f"Pipe diameters: {diameters}")
        
        demands = []
        for node, node_data in network.junctions():
            demand = node_data.base_demand if node_data.base_demand is not None else 0.0
            demands.append(demand)
        
        # print(f"Node demands: {demands}")
        
        # Get pressures from dictionary instead of node objects
        pressures = []
        for node, node_data in network.junctions():
            node_name = node_data.name
            # Get pressure from dictionary, default to 0.0 if not found
            pressure = self.node_pressures.get(node_name, 0.0)
            pressures.append(pressure)
        
        # print(f"Node pressures: {pressures}")
        
        # Global features of current pipe ID
        current_pipe_idx = self.current_pipe_index if self.pipe_names else 0
        
        # Add all features to the list
        features.extend(diameters)
        features.extend(demands)
        features.extend(pressures)
        features.append(float(current_pipe_idx))
        
        # Ensure all features are numerical
        numeric_features = []
        for feat in features:
            if feat is None:
                numeric_features.append(0.0)
            else:
                try:
                    numeric_features.append(float(feat))
                except (ValueError, TypeError):
                    numeric_features.append(0.0)
        
        return np.array(numeric_features, dtype=np.float32) # 32 bit float for quick calc
        
    def simulate_network(self, network: wntr.network.WaterNetworkModel):

        """Run the EPANET simulation for the current network."""
        try:
            results = run_epanet_simulation(network)
            performance_metrics = evaluate_network_performance(network, results)
            return results, performance_metrics
        except Exception as e:
            print(f"Simulation failed: {e}")
            return None
        
    def reward(self, original_pipe_diameters, actions, performance_metrics, downgraded_pipes, disconnections, actions_causing_disconnections):
        """Calculate the reward based on performance metrics and topology changes."""
        # Create a simple structure with the original pipe diameters and current network
        # to pass to calculate_reward
        reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, downgraded_pipes = calculate_reward(
            self.current_network,  # Current network state
            original_pipe_diameters,  # Original diameters
            actions,
            self.pipes,
            performance_metrics,
            self.labour_cost,
            downgraded_pipes,
            disconnections,
            actions_causing_disconnections
        )

        return reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, downgraded_pipes
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        # Select random scenario
        self.current_scenario = random.choice(self.scenarios)
        print(f"Resetting environment for scenario: {self.current_scenario}")
        self.network_states = self.load_network_states(self.current_scenario)
        print(f"Loaded {len(self.network_states)} network states for scenario: {self.current_scenario}")
        
        # Reset episode state
        self.current_time_step = 0
        self.current_pipe_index = 0
        
        # Load initial network
        initial_network_path = self.network_states[0]
        self.current_network = wntr.network.WaterNetworkModel(initial_network_path)
        self.pipe_names = list(self.current_network.pipe_name_list)
        # print(f"Pipes in the initial network: {self.pipe_names}")

        self.node_pressures = {}

        results, performance_metrics = self.simulate_network(self.current_network)

        # Store pressures in dictionary instead of on node objects
        if results is not None:
            for node_name in self.current_network.junction_name_list:
                pressure = np.mean(results.node['pressure'][node_name])
                self.node_pressures[node_name] = pressure
        
        # Set observation space if not already set
        if self.observation_space is None:
            sample_obs = self.get_network_features(self.current_network)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
            )
        
        self.episode_count += 1
        
        return self.get_network_features(self.current_network)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Start time
        start_time = time.time()

        # Store only the original network topology as a graph
        original_graph = self.current_network.to_graph().to_undirected()
        
        # Store original pipe diameters
        original_pipe_diameters = {}
        for pipe_name in self.pipe_names:
            pipe = self.current_network.get_link(pipe_name)
            original_pipe_diameters[pipe_name] = pipe.diameter
        
        # actions = []
        downgraded_pipes = False
        
        # Process the action
        if action > 0:  # Change pipe diameter
            new_diameter = self.pipe_diameter_options[action - 1]
            pipe_name = self.pipe_names[self.current_pipe_index]
            pipe = self.current_network.get_link(pipe_name)
            old_diameter = pipe.diameter

            if new_diameter < old_diameter:
                # Only allow downgrades
                downgraded_pipes = True
                print(f"Downgrading pipe {pipe_name} from diameter {old_diameter} to {new_diameter}")

            pipe.diameter = new_diameter
            self.actions.append((pipe_name, new_diameter))
            print(f"Action taken: {action}, pipe {pipe_name} changed from diameter {old_diameter} to {new_diameter}")
        else:
            print(f"No change for pipe {self.pipe_names[self.current_pipe_index]} (index {self.current_pipe_index})")
        
        # Move to next pipe
        self.current_pipe_index += 1
        
        # Check if all pipes processed for current time step
        if self.current_pipe_index >= len(self.pipe_names):
            # Run simulation once and store results
            self.results, performance_metrics = self.simulate_network(self.current_network)
            
            # Update pressures in the network object for feature extraction later
            pressure_data = []
            if self.results is not None:
                for node_name in self.current_network.junction_name_list:
                    pressure = np.mean(self.results.node['pressure'][node_name])
                    node = self.current_network.get_node(node_name)
                    # Store pressure in the node object for later access
                    pressure_data.append(pressure)
            
            # Check for disconnections without using a full copy of the network
            disconnections, actions_causing_disconnections = self.check_disconnections(
                original_graph, self.actions)
            
            # Calculate reward using only necessary data
            reward, cost, pd_ratio, demand_satisfaction, disconnections, actions_causing_disconnections, downgraded = self.reward(
                original_pipe_diameters,
                self.actions,
                performance_metrics,
                downgraded_pipes,
                disconnections,
                actions_causing_disconnections
            )

            # Move to next time step
            self.current_time_step += 1
            self.current_pipe_index = 0
            
            # Load next network state if available
            if self.current_time_step < len(self.network_states):
                next_network_path = self.network_states[self.current_time_step]
                # Update network with evolved topology but keep optimized diameters
                temp_network = wntr.network.WaterNetworkModel(next_network_path)
                
                # Transfer new diameters to new network topology
                for pipe_name in self.pipe_names:
                    if pipe_name in temp_network.pipe_name_list:
                        old_pipe = self.current_network.get_link(pipe_name)
                        new_pipe = temp_network.get_link(pipe_name)
                        new_pipe.diameter = old_pipe.diameter
                
                self.current_network = temp_network
                self.pipe_names = list(self.current_network.pipe_name_list)

                # Increase the roughness of the pipes by H-W roughness coefficient X - how much does roughness increase in pipes every 6 months?
                for pipe, pipe_data in self.current_network.pipes():
                    pipe_id = pipe[0]
                    roughness = self.current_network.get_link(pipe_id).roughness
                    pipe_data.roughness -= 0.025
                    """H-W coefficient decrease by 0.025 every 6 months, or 2.5 decade"""

                # Add a small random probability of pipe burst

                # Reset actions
                self.actions = []
        else:
            reward = 0  # No reward until time step is complete
        
        # Check if episode is done
        done = self.current_time_step >= len(self.network_states)
        
        # Get next observation
        obs = self.get_network_features(self.current_network)

        end_time = time.time()
        
        # Info dictionary
        info = {
            'scenario': self.current_scenario,
            'time_step': self.current_time_step,
            'pipe_index': self.current_pipe_index,
            'total_pipes': len(self.pipe_names),
            'episode': self.episode_count,
            'action': action,
            'reward': reward,
            'step_time': end_time - start_time,
        }
        
        return obs, reward, done, info
    
    def check_disconnections(self, original_graph, actions):
        """Check if any actions would cause network disconnections."""
        disconnections = False
        actions_causing_disconnections = []
        
        # Current network graph
        current_graph = self.current_network.to_graph().to_undirected()
        
        # Check if all nodes are still connected
        if nx.number_connected_components(current_graph) > nx.number_connected_components(original_graph):
            disconnections = True
            # Identify which actions caused disconnections
            for action in actions:
                pipe_name, new_diameter = action
                # You can implement logic here to identify problematic actions
                # For example, by checking each pipe removal individually
                actions_causing_disconnections.append(action)
        
        return disconnections, actions_causing_disconnections
    
    def get_action_mask(self) -> np.ndarray:
        """
        Create an action mask where invalid actions (smaller pipe diameters) are masked out.
        Returns a boolean array where True means the action is valid.
        """
        # Get current pipe diameter
        current_pipe_name = self.pipe_names[self.current_pipe_index]
        current_pipe = self.current_network.get_link(current_pipe_name)
        current_diameter = current_pipe.diameter
        
        # Initialize mask with all actions as valid
        action_mask = np.ones(self.action_space.n, dtype=bool)
        
        # Action 0 (no change) is always valid
        # For other actions, check if the new diameter would be smaller
        for i in range(1, self.action_space.n):
            new_diameter = self.pipe_diameter_options[i-1]
            if new_diameter < current_diameter:
                action_mask[i] = False
        
        return action_mask
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        print(f"Scenario: {self.current_scenario}, Time Step: {self.current_time_step}, Pipe Index: {self.current_pipe_index}")
        print(f"Current Network: {self.current_network.name}")
        print(f"Current Pipe Diameters: {[pipe.diameter for pipe in self.current_network.pipes()]}")

    def close(self):
        """Close the environment (optional)."""
        pass

# Example usage:
# Example usage:
if __name__ == "__main__":

    start_episode_time = time.time()
    
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    
    scenarios = [
        'anytown_densifying_1', 
        'anytown_densifying_2',
        'anytown_densifying_3',
        'anytown_sprawling_1',
        'anytown_sprawling_2',
        'anytown_sprawling_3',
        'hanoi_densifying_1',
        'hanoi_densifying_2',
        'hanoi_densifying_3',
        'hanoi_sprawling_1',
        'hanoi_sprawling_2',
        'hanoi_sprawling_3'
    ]

    def select_no_downgrades(env):
        action_mask = env.get_action_mask()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        return random.choice(valid_actions) if valid_actions else 0  # Default to no change if no valid actions

    env = WNTRGymEnv(pipes, scenarios)
    obs = env.reset()
    # print("Initial Observation:", obs)
    
    network = env.current_network
    # for pipe, pipe_data in network.pipes():
    #     print(f"Pipes in initial network: {pipe}, Diameter: {pipe_data.diameter}, Roughness:{pipe_data.roughness}")
    
    # Run through all time steps
    time_step = 0
    total_reward = 0

    rewards= []
    
    while True:
        # Process all pipes for the current time step
        print(f"\n===== Time Step {time_step} =====")
        print(f"Scenario: {env.current_scenario}")
        pipes_in_current_step = len(env.pipe_names)
        
        for pipe_idx in range(pipes_in_current_step):
            # action = env.action_space.sample()  # Random action
            action = select_no_downgrades(env)  # Select an action with no downgrades
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            # total_reward += reward
            
            # print(f"Pipe {pipe_idx+1}/{pipes_in_current_step}, Action: {action}, Reward: {reward}")
            
            if done:
                print(f"\nEpisode finished")
                end_episode_time = time.time()
                print(f"Episode duration: {end_episode_time - start_episode_time:.2f} seconds")
                env.close()
                exit()

                # Plot the reward with demand increase
        
        # After processing all pipes in a time step
        time_step += 1

        new_network = env.current_network
        # for pipe, pipe_data in new_network.pipes():
        #     print(f"Pipe {pipe}: Diameter = {pipe_data.diameter}, Roughness = {pipe_data.roughness}")

        # print(f"Completed time step {time_step-1}. Cumulative reward: {total_reward}")
        

    

