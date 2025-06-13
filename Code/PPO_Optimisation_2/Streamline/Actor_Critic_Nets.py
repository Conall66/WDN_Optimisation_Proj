import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch, Data

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from PPO_Env import PPOEnv  # Assuming your environment is in PPO_Env.py

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network feature extractor for the water network environment.
    This class processes the dictionary observation from the environment and outputs
    a single feature vector for the SB3 policy and value networks.

    :param observation_space: The observation space of the environment.
    :param features_dim: The desired output dimension of the feature extractor.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Extract dimensions from the observation space ---
        node_features_dim = observation_space['node_features'].shape[0]
        edge_features_dim = observation_space['edge_features'].shape[0]
        global_state_dim = observation_space['global_state'].shape[0]

        gnn_embedding_dim = 128  # The output size of the GNN layers

        # --- Define GNN layers ---
        # We use GATv2Conv as it's a powerful and expressive graph convolution layer.
        self.gnn_layer_1 = GATv2Conv(node_features_dim, gnn_embedding_dim, edge_dim=edge_features_dim)
        self.gnn_layer_2 = GATv2Conv(gnn_embedding_dim, gnn_embedding_dim, edge_dim=edge_features_dim)

        # --- Define the post-processing linear layer ---
        # This layer will combine the GNN output with other state information.
        # It takes as input:
        # 1. Global graph embedding (gnn_embedding_dim)
        # 2. Start node embedding (gnn_embedding_dim)
        # 3. End node embedding (gnn_embedding_dim)
        # 4. Global state vector (global_state_dim)
        combined_input_dim = gnn_embedding_dim * 3 + global_state_dim
        self.linear = nn.Linear(combined_input_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: dict) -> torch.Tensor:
        device = self.gnn_layer_1.lin_l.weight.device

        # 1. Convert list of GraphInstances to a list of PyG Data objects
        # `observations['graph']` is now a list of GraphInstance objects.
        data_list = []
        for graph_instance in observations['graph']:
            # Convert the Gymnasium GraphInstance back to a PyG Data object
            data = Data(
                x=torch.from_numpy(graph_instance.nodes).to(dtype=torch.float32),
                # Transpose edge_links [num_edges, 2] back to PyG's edge_index [2, num_edges]
                edge_index=torch.from_numpy(graph_instance.edge_links).to(dtype=torch.long).t().contiguous(),
                edge_attr=torch.from_numpy(graph_instance.edges).to(dtype=torch.float32)
            )
            data_list.append(data)

        # 2. Collate the list of Data objects into a single PyG Batch
        graph_batch = Batch.from_data_list(data_list).to(device)

        # --- From here, the logic is identical to the previous "no padding" solution ---

        # 3. GNN Pass
        x = self.relu(self.gnn_layer_1(graph_batch.x, graph_batch.edge_index, edge_features=graph_batch.edge_attr))
        x = self.relu(self.gnn_layer_2(x, graph_batch.edge_index, edge_features=graph_batch.edge_attr))

        # 4. Aggregate Features
        global_graph_embedding = global_mean_pool(x, graph_batch.batch)
        
        # Other features need to be moved to the correct device
        global_state = observations['global_state'].to(device)
        current_pipe_nodes = observations['current_pipe_nodes'].to(device)

        # 5. Get embeddings for specific pipe nodes
        node_offsets = graph_batch.ptr[:-1]
        start_node_indices = current_pipe_nodes[:, 0] + node_offsets
        end_node_indices = current_pipe_nodes[:, 1] + node_offsets
        pipe_node_embeddings = torch.cat([x[start_node_indices], x[end_node_indices]], dim=1)

        # 6. Concatenate and process
        combined_features = torch.cat([global_graph_embedding, pipe_node_embeddings, global_state], dim=1)
        return self.relu(self.linear(combined_features))

if __name__ == '__main__':

    # --- 1. Instantiate the Environment ---
    print("initialising environment...")
    network_dir = 'Networks/Simple_Nets'
    env = PPOEnv(network_files_dir=network_dir)
    print("Environment initialised.")

    # --- 2. Define Policy Keywords ---
    # This dictionary tells the PPO agent to use our custom GNN feature extractor.
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256), # Output features from GNN extractor
        net_arch=dict(pi=[128, 64], vf=[128, 64]) # Actor-critic network architecture

    )

    # --- 3. Instantiate the PPO Agent ---
    print("Creating PPO agent with GNN policy...")
    model = PPO(
        policy="MultiInputPolicy",  # Use MultiInputPolicy for dictionary observations
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps to run for each environment per update
        batch_size=64,  # Batch size for each gradient update
        n_epochs=10,  # Number of epochs to update the policy
        gamma=0.9,  # Discount factor
        gae_lambda=0.95,  # GAE lambda for advantage estimation
        clip_range=0.2,  # Clipping range for PPO
        ent_coef=0.01,  # Coefficient for the entropy term
        vf_coef=0.5,  # Coefficient for the value function loss
        max_grad_norm=0.5,  # Maximum gradient norm for clipping
        verbose=2,
    )
    print("Agent created.")

    # --- 4. Train the Agent ---
    print("Starting model training...")
    model.learn(total_timesteps=20000) # Use a larger number for real training
    # --- 5. Save the Trained Model ---
    print("Training finished. Saving model...")
    model.save("ppo_gnn_wntr_model")
    print("Model saved to ppo_gnn_wntr_model.zip")

    # To load and use the model later:
    # loaded_model = PPO.load("ppo_gnn_wntr_model")
    # obs, _ = env.reset()
    # action, _ = loaded_model.predict(obs, deterministic=True)