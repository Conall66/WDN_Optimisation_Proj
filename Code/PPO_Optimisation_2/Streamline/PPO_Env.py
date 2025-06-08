
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

import gymnasium as gym
from gymnasium import spaces

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
            network_files_dir='Networks/Simple_Nets',):
        """
        Initialise the environment with a water network model loaded from an .inp file.

        Parameters:
        inp_file (str): Path to the .inp file containing the water network model.
        """
        super(PPOEnv, self).__init__()

        # File management intialisation
        self.network_files_dir = network_files_dir # Change this to the directory of saved networks
        self.network_files = sorted(os.listdir(self.network_files_dir))

        # Load the water network model from the .inp file
        self.initial_wn = WaterNetworkModel(self.network_files[0])
        self.wn = None # This is updated with each step, and set to initial network in the reset

        self.budget = 0 # This will be rewritten in the reset function
        self.energy_cost = 0.26 # £0.26 per kWh, this is the cost of energy used in the simulation
        self.min_pressure_m = 0 # Minimum pressure in meters, this is the minimum pressure required at each junction
        self.pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
        } # These are the pipes with diameters and associated unit costs the network can be upgraded/downgraded

        self.current_pipe_index = 0 # This is the index of the current pipe being modified by the agent
        self.results = None # This will be updated with the results of the hydraulic simulation after each step
    
        # Define action and observation space. 0 is an action of not changing the pipe diameter, and 1-N is the action of inceasing the diameter to N of the pipes dictionary
        self.action_space = spaces.Discrete(len(self.pipes) + 1)

        max_nodes = 200
        max_edges = 300

        self.observation_space = gym.spaces.Dict({
            'node_features': spaces.Box(low=0, high=1, shape=(max_nodes, 2), dtype=np.float32), # Pressures and demands
            'edge_features': spaces.Box(low=0, high=1, shape=(max_edges, 2), dtype=np.float32), # Pipe diameters, is_current
            'edge_index': spaces.Box(low=0, high=max_nodes-1, shape=(2, max_edges), dtype=np.int64), # Edge indices for graph representation (bidirectional)
            'budget': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def _get_observation(self):

        """
        Construct a graph based representation of the water network for the GNN in the actor-critic networks. 
        """

        pressures = self.results.node['pressure'].loc[self.results.node['pressure'].index.max()]
        demands = self.results.node['demand'].loc[self.results.node['demand'].index.max()]
        
        node_map = {name: i for i, name in enumerate(self.wn.node_name_list)}
        
        node_features = np.zeros((len(node_map), 2), dtype=np.float32)
        for name, i in node_map.items():
            if name in pressures.index:
                node_features[i, 0] = pressures[name]
            if name in demands.index:
                node_features[i, 1] = demands[name]

        start_nodes, end_nodes, edge_features = [], [], []
        
        # Get the name of the pipe currently being considered for an action
        current_pipe_name = self.wn.pipe_name_list[self.current_pipe_index]

        for pipe_name, pipe in self.wn.pipes():
            start_node_idx = node_map[pipe.start_node_name]
            end_node_idx = node_map[pipe.end_node_name]
            start_nodes.extend([start_node_idx, end_node_idx])
            end_nodes.extend([end_node_idx, start_node_idx])
            
            # Highlight the current pipe in the edge features
            is_current = 1.0 if pipe_name == current_pipe_name else 0.0
            features = [pipe.diameter, pipe.length, is_current]
            edge_features.extend([features, features])

        edge_index = np.array([start_nodes, end_nodes], dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32)

        # Pad observations to match the predefined max size
        num_nodes = len(node_map)
        num_edges = edge_index.shape[1]
        padded_node_features = np.pad(node_features, ((0, self.observation_space['node_features'].shape[0] - num_nodes), (0, 0)), 'constant')
        padded_edge_index = np.pad(edge_index, ((0, 0), (0, self.observation_space['edge_index'].shape[1] - num_edges)), 'constant')
        padded_edge_features = np.pad(edge_features, ((0, self.observation_space['edge_features'].shape[0] - num_edges), (0, 0)), 'constant')

        return {
            'node_features': padded_node_features.astype(np.float32),
            'edge_index': padded_edge_index.astype(np.int64),
            'edge_features': padded_edge_features.astype(np.float32)
        }
    
    
    def _calc_reward(self):

        """
        Calculate the reward based on a weightings for normalised cost, pressure and demand satisfaction.
        """

        # The cost is the sum of upgrading and downgrading pipes (unit cost of pipe itself plus labour costs) and energy usage in the network. All variable costs will be attributed to step cost

        cost = self.change_costs + self.energy_cost
        
        pressures = self.results.node['pressure'].loc[self.results.node['pressure'].index.max()]
        pressure_deficit = np.sum(np.maximum(0, self.min_pressure_m - pressures.loc[self.wn.junction_name_list]))
        pressure_penalty = -1.0 * pressure_deficit
    
