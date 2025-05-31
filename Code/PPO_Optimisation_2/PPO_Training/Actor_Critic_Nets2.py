"""
Graph Neural Network Actor-Critic Networks for Water Distribution Network Optimization

This module implements GNN-based actor and critic networks that can handle 
variable-sized water distribution networks for PPO-based pipe sizing optimization.

Requirements:
- torch
- torch_geometric
- stable-baselines3
- networkx
- numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import gymnasium as gym  # Changed from gym to gymnasium (more up to date version)
from gymnasium import spaces

class WaterNetworkGraphConverter:
    """
    Converts water distribution network to PyTorch Geometric format
    """
    
    def __init__(self, pipes_config: Dict):
        """
        Args:
            pipes_config: Dictionary with pipe diameter options and costs
        """
        self.pipes_config = pipes_config
        self.pipe_diameter_options = [pipe_data['diameter'] for pipe_data in pipes_config.values()]
        
    def network_to_graph_data(self, network, node_pressures: Dict, current_pipe_index: int = 0) -> Data:
        """
        Convert WNTR network to PyTorch Geometric Data object
        
        Args:
            network: WNTR WaterNetworkModel
            node_pressures: Dictionary of node pressures
            current_pipe_index: Index of current pipe being optimized
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract nodes (junctions and reservoirs)
        nodes = []
        node_features = []
        node_name_to_idx = {}
        
        # Add junctions
        for i, (node_name, node_data) in enumerate(network.junctions()):
            nodes.append(node_name)
            node_name_to_idx[node_name] = i
            
            # Node features: [demand, elevation, pressure, is_junction]
            demand = node_data.base_demand if node_data.base_demand is not None else 0.0
            elevation = node_data.elevation if hasattr(node_data, 'elevation') else 0.0
            pressure = node_pressures.get(node_name, 0.0)
            is_junction = 1.0
            
            node_features.append([demand, elevation, pressure, is_junction])
        
        # Add reservoirs
        for node_name, node_data in network.reservoirs():
            if node_name not in node_name_to_idx:
                idx = len(nodes)
                nodes.append(node_name)
                node_name_to_idx[node_name] = idx
                
                # Reservoir features: [0, base_head, base_head, is_reservoir]
                base_head = node_data.base_head if hasattr(node_data, 'base_head') else 0.0
                is_junction = 0.0
                
                node_features.append([0.0, base_head, base_head, is_junction])
        
        # Extract edges (pipes) and edge features
        edge_index = []
        edge_features = []
        pipe_names = []
        
        for pipe_name, pipe_data in network.pipes():
            start_node = pipe_data.start_node_name
            end_node = pipe_data.end_node_name
            
            if start_node in node_name_to_idx and end_node in node_name_to_idx:
                start_idx = node_name_to_idx[start_node]
                end_idx = node_name_to_idx[end_node]
                
                # Add bidirectional edges
                edge_index.extend([[start_idx, end_idx], [end_idx, start_idx]])
                
                # Edge features: [diameter, length, roughness, is_current_pipe]
                diameter = pipe_data.diameter
                length = pipe_data.length
                roughness = pipe_data.roughness
                is_current_pipe = 1.0 if len(pipe_names) == current_pipe_index else 0.0
                
                edge_feature = [diameter, length, roughness, is_current_pipe]
                edge_features.extend([edge_feature, edge_feature])  # Same features for both directions
                
                pipe_names.append(pipe_name)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Add global features
        global_features = torch.tensor([
            len(nodes),  # Number of nodes
            len(pipe_names),  # Number of pipes
            current_pipe_index,  # Current pipe index
        ], dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   global_features=global_features, pipe_names=pipe_names)

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using GNN for variable-sized water networks
    """
    
    def __init__(self, observation_space: gym.Space, pipes_config: Dict, 
                 hidden_dim: int = 64, num_layers: int = 3):
        """
        Args:
            observation_space: Gym observation space
            pipes_config: Dictionary with pipe diameter options and costs
            hidden_dim: Hidden dimension size for GNN layers
            num_layers: Number of GNN layers
        """
        # Feature dimension will be determined by the GNN output
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim=hidden_dim)
        
        self.pipes_config = pipes_config
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_converter = WaterNetworkGraphConverter(pipes_config)
        
        # Node feature dimension: [demand, elevation, pressure, is_junction]
        node_input_dim = 4
        # Edge feature dimension: [diameter, length, roughness, is_current_pipe]
        edge_input_dim = 4
        # Global feature dimension: [num_nodes, num_pipes, current_pipe_index]
        global_input_dim = 3
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        
        # Edge embedding layers
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Global feature processing
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final feature combination
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Graph features + global features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN feature extractor
        
        Args:
            observations: Tensor observations from environment
            
        Returns:
            Extracted features tensor
        """
        # Handle different input formats
        if isinstance(observations, dict):
            # Dictionary observations (from GraphAwareWNTREnv)
            return self._process_dict_observations(observations)
        else:
            # Flattened observations (from base environment)
            return self._process_flattened_observations(observations)
    
    def _process_dict_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process dictionary-based observations"""
        batch_size = observations['nodes'].shape[0]
        features = []
        
        for i in range(batch_size):
            # Extract components from the dictionary
            nodes = observations['nodes'][i]
            edges = observations['edges'][i]
            globals_data = observations['globals'][i]
            
            # Process non-zero nodes
            node_mask = (nodes.sum(dim=1) != 0)
            if node_mask.sum() > 0:
                valid_nodes = nodes[node_mask]
                node_features = F.relu(self.node_embedding(valid_nodes))
                graph_features = torch.mean(node_features, dim=0)
            else:
                graph_features = torch.zeros(self.hidden_dim, device=nodes.device)
            
            # Process global features
            global_features = F.relu(self.global_mlp(globals_data))
            
            # Combine features
            combined_features = torch.cat([graph_features, global_features])
            feature = self.final_mlp(combined_features)
            features.append(feature)
        
        return torch.stack(features)
    
    def _process_flattened_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process flattened observation (fallback when graph data isn't directly available)
        """
        batch_size = observations.shape[0]
        features = []
        
        for i in range(batch_size):
            obs = observations[i]
            
            # Simple processing for flattened observations
            # Assume the first few elements are node features
            if len(obs) >= 4:
                # Use first 4 elements as representative node features
                node_input = obs[:4].unsqueeze(0)
                node_features = F.relu(self.node_embedding(node_input))
                graph_features = node_features.flatten()
            else:
                graph_features = torch.zeros(self.hidden_dim, device=obs.device)
            
            # Create dummy global features if not available
            global_input = torch.zeros(3, device=obs.device)
            if len(obs) > 4:
                # Use some elements from the observation as global features
                available_elements = min(3, len(obs) - 4)
                global_input[:available_elements] = obs[4:4+available_elements]
            
            global_features = F.relu(self.global_mlp(global_input))
            
            # Combine features
            if graph_features.shape[0] != self.hidden_dim:
                # Pad or truncate to match expected size
                if graph_features.shape[0] < self.hidden_dim:
                    padding = torch.zeros(self.hidden_dim - graph_features.shape[0], device=obs.device)
                    graph_features = torch.cat([graph_features, padding])
                else:
                    graph_features = graph_features[:self.hidden_dim]
            
            combined_features = torch.cat([graph_features, global_features])
            feature = self.final_mlp(combined_features)
            features.append(feature)
        
        return torch.stack(features)
    
    def process_graph_data(self, graph_data: Data) -> torch.Tensor:
        """
        Process actual graph data through GNN
        
        Args:
            graph_data: PyTorch Geometric Data object
            
        Returns:
            Graph-level features
        """
        # Node embeddings
        x = F.relu(self.node_embedding(graph_data.x))
        
        # GNN forward pass
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, graph_data.edge_index))
        
        # Global pooling to get graph-level representation
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # Single graph
        graph_features = self.global_pool(x, batch)
        
        # Process global features
        global_features = F.relu(self.global_mlp(graph_data.global_features))
        
        # Combine features
        combined_features = torch.cat([graph_features, global_features])
        final_features = self.final_mlp(combined_features)
        
        return final_features

class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy using GNN feature extraction
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, pipes_config, **kwargs):
        self.pipes_config = pipes_config
        super(GNNActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, 
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs={"pipes_config": pipes_config},
            **kwargs
        )

class GraphPPOAgent:
    """
    PPO Agent with GNN-based policy for water distribution network optimization
    
    """
    
    def __init__(self, env, pipes_config: Dict, **ppo_kwargs):
        """
        Args:
            env: Water network gym environment
            pipes_config: Dictionary with pipe diameter options and costs
            **ppo_kwargs: Additional arguments for PPO
        """
        self.env = env
        self.pipes_config = pipes_config
        self.graph_converter = WaterNetworkGraphConverter(pipes_config)

        device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU to accelerate training if available
        print(f"Using device: {device}")
        
        # Default PPO parameters
        default_ppo_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 2
        }

        default_ppo_kwargs.update(ppo_kwargs)
        
        # Create PPO agent with custom policy
        self.agent = PPO(
            GNNActorCriticPolicy,
            env,
            policy_kwargs={"pipes_config": pipes_config},
            device = device,
            **default_ppo_kwargs
        )
    
    def train(self, total_timesteps: int):
        """Train the agent"""
        self.agent.learn(total_timesteps=total_timesteps)
    
    def predict(self, observation, deterministic: bool = True):
        """Make prediction"""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the trained model"""
        self.agent.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.agent.load(path)

# Enhanced Environment Wrapper for Graph Data
class GraphAwareWNTREnv(gym.Wrapper):
    """
    Wrapper for WNTR environment that provides graph-aware functionality

    THE FUNCTIONALISTY OF THIS CLASS IS NOW BUILT INTO THE WNTRGYMENV CLASS AND IS NO LONGER IN USE

    """
    
    def __init__(self, env, pipes_config):
        super(GraphAwareWNTREnv, self).__init__(env)
        self.pipes_config = pipes_config
        self.graph_converter = WaterNetworkGraphConverter(pipes_config)
        self.current_graph_data = None
        self.use_graph_obs = False  # Flag to control observation type

        # Keep the original observation space for compatibility
        self.observation_space = env.observation_space
        
        print("GraphAwareWNTREnv initialized with observation space:", self.observation_space)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._update_graph_data()
        return obs  # Return original observation for now
    
    def step(self, action):
        obs, reward, terminated, done, info = self.env.step(action)
        self._update_graph_data()
        return obs, reward, terminated, done, info  # Return original observation for now
    
    def _update_graph_data(self):
        # Convert current network to graph data
        try:
            network = self.env.current_network
            node_pressures = getattr(self.env, 'node_pressures', {})
            current_pipe_index = getattr(self.env, 'current_pipe_index', 0)
            
            self.current_graph_data = self.graph_converter.network_to_graph_data(
                network, node_pressures, current_pipe_index
            )
        except Exception as e:
            print(f"Warning: Could not update graph data: {e}")
            self.current_graph_data = None
    
    def get_graph_data(self) -> Optional[Data]:
        """Get current graph data"""
        return self.current_graph_data
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for current pipe
        """
        if hasattr(self.env, 'get_action_mask'):
            return self.env.get_action_mask()
        
        return np.ones(self.action_space.n, dtype=bool)

# Example usage and training script
def create_and_train_gnn_agent():
    """
    Example function showing how to create and train the GNN-based PPO agent
    """
    
    # Pipe configuration (from your original script)
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }
    
    scenarios = [
        'anytown_densifying_1', 'anytown_densifying_2', 'anytown_densifying_3',
        'anytown_sprawling_1', 'anytown_sprawling_2', 'anytown_sprawling_3',
        'hanoi_densifying_1', 'hanoi_densifying_2', 'hanoi_densifying_3',
        'hanoi_sprawling_1', 'hanoi_sprawling_2', 'hanoi_sprawling_3'
    ]
    
    # Import your environment
    from PPO_Environment import WNTRGymEnv
    import os

    agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
    os.makedirs(agents_dir, exist_ok=True)
    
    # Create environment
    base_env = WNTRGymEnv(pipes, scenarios)
    env = GraphAwareWNTREnv(base_env, pipes)
    env.reset()

    print("Environment created with observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Create and train agent
    agent = GraphPPOAgent(env, pipes, verbose=2)
    agent.train(total_timesteps=50000)
    
    # Save the trained agent
    model_path = os.path.join(agents_dir, "gnn_ppo_water_network")
    agent.save(model_path)

    print("GNN-based PPO agent setup complete!")

if __name__ == "__main__":
    create_and_train_gnn_agent()