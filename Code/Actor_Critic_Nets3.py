"""
Graph Neural Network Actor-Critic Networks for Water Distribution Network Optimization

This module implements GNN-based actor and critic networks that can handle 
variable-sized water distribution networks for PPO-based pipe sizing optimization.

Requirements:
- torch
- torch_geometric
- stable-baselines3
- gymnasium
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import gymnasium as gym

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Graph Convolutional Network (GCN).
    It intelligently handles padded observations from the environment by
    reconstructing the original, variable-sized graphs before processing.
    This ensures maximum efficiency by not performing computations on zero-padding.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, hidden_dim: int = 64, num_layers: int = 3):
        """
        Args:
            observation_space: The dictionary observation space from the Gym environment.
            hidden_dim: The size of the hidden layers in the GNN and MLPs.
            num_layers: The number of GCN layers for message passing.
        """
        super().__init__(observation_space, features_dim=hidden_dim)
        
        self.hidden_dim = hidden_dim
        
        # Define dimensions from the observation space's fixed-size shape
        node_input_dim = observation_space["nodes"].shape[1]
        edge_input_dim = observation_space["edges"].shape[1]
        global_input_dim = observation_space["globals"].shape[0]
        
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.global_pool = global_mean_pool
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass that takes padded observations, removes the padding,
        and processes the real graph data efficiently.
        """
        graph_data_list = []
        # The 'globals' feature vector contains the true number of nodes and pipes
        # at indices 0 and 1, respectively.
        true_num_nodes_list = observations['globals'][:, 0].int()
        true_num_pipes_list = observations['globals'][:, 1].int()
        
        # Iterate over each graph in the batch from the SB3 environment
        for i in range(observations['nodes'].shape[0]):
            num_nodes = true_num_nodes_list[i]
            num_pipes = true_num_pipes_list[i]
            
            # Slice the padded tensors to get only the real data
            nodes_data = observations['nodes'][i][:num_nodes]
            edges_data = observations['edges'][i][:num_pipes]
            
            # The edge_index needs to be filtered to only include connections
            # between existing nodes.
            edge_index_full = observations['edge_index'][i].long()
            # A valid edge must have both start and end nodes within the true node count
            valid_edge_mask = (edge_index_full[0, :] < num_nodes) & (edge_index_full[1, :] < num_nodes)
            edge_index_data = edge_index_full[:, valid_edge_mask]
            
            # Create a PyG Data object with the *un-padded*, real data
            graph_data = Data(
                x=nodes_data,
                edge_index=edge_index_data,
                edge_attr=edges_data, # This is not used by GCNConv but is good practice
                global_features=observations['globals'][i]
            )
            graph_data_list.append(graph_data)

        # `Batch.from_data_list` handles the different-sized graphs automatically
        batch = Batch.from_data_list(graph_data_list).to(self.device)
        
        # --- Standard GNN processing on the clean, batched data ---
        x = F.relu(self.node_embedding(batch.x))
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, batch.edge_index))
        
        # `batch.batch` correctly maps nodes to their original graph in the batch
        graph_features = self.global_pool(x, batch.batch)

        # Process global features directly from original observation
        global_features_processed = F.relu(self.global_mlp(observations['globals']))
        
        # Now both tensors should have compatible shapes
        combined_features = torch.cat([graph_features, global_features_processed], dim=1)
        final_features = self.final_mlp(combined_features)
        
        return final_features

class GNNActorCriticPolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy that uses the GNNFeatureExtractor."""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space, action_space, lr_schedule, 
            features_extractor_class=GNNFeatureExtractor,
            **kwargs
        )

class GraphPPOAgent:
    """A wrapper class for the Stable Baselines 3 PPO agent to simplify creation."""
    def __init__(self, env, **ppo_kwargs):
        self.env = env
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        default_ppo_kwargs = {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "verbose": 1, "device": device
        }
        default_ppo_kwargs.update(ppo_kwargs)
        
        self.agent = PPO(GNNActorCriticPolicy, env, **default_ppo_kwargs)
    
    def train(self, total_timesteps: int, callback=None):
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, observation, deterministic: bool = True):
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        self.agent.save(path)
    
    def load(self, path: str, env=None):
        self.agent = PPO.load(path, env=env)

# Test script
if __name__ == "__main__":
    # Example usage
    pipes_config = {'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
                    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32}, 
                    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
                    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47}, 
                    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
                    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}}
    
    # Create a dummy observation space
    observation_space = gym.spaces.Dict({
        "nodes": gym.spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32),
        "edges": gym.spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32),
        "edge_index": gym.spaces.Box(low=0, high=10, shape=(2, 20), dtype=np.int32),
        "globals": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    })
    
    feature_extractor = GNNFeatureExtractor(observation_space)
    print("Feature extractor initialized with output dimension:", feature_extractor.features_dim)