
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
import wandb

from PPO_Env import PPOEnv

"""
Movinf away from stable-baselines to CleanRL for better integration with GNNs and custom environments.
"""

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanrl-gnn-wntr"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "PPOEnv"
    total_timesteps: int = 50000 # Just checking we run to completion
    learning_rate: float = 3e-4
    num_envs: int = 1 # CleanRL can run vectorized envs, but we start with 1 for simplicity
    num_steps: int = 431 # Steps per policy rollout
    anneal_lr: bool = True
    gamma: float = 0.9
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 5
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

def make_env(env_id, seed, network_dir):
    def thunk():
        env = PPOEnv(network_files_dir=network_dir)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# In Actor_Critic_CleanRL.py, inside the RolloutBuffer class

class RolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # We will just store the raw observation dictionaries
        self.observations = [None] * buffer_size
        
        # Storage for other PPO components
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, done, value, log_prob):
        """Add a new experience to the buffer"""
        if self.ptr < self.buffer_size:
            self.observations[self.ptr] = obs # Store the dict
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.ptr += 1
            self.size = max(self.size, self.ptr)

    def compute_returns_and_advantages(self, last_value):
        # This method remains unchanged
        # ... (keep existing code)
        pass

    def get_batches(self, batch_size):
        """Get batches of data for training"""
        indices = np.random.permutation(self.size) # Use self.size
        for start_idx in range(0, self.size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Extract observation components from the stored list of dicts
            batch_obs_list = [self.observations[i] for i in batch_indices]

            # print(f"Global State: {[obs['global_state'] for obs in batch_obs_list]}")
            
            # Collate the batch
            batch_obs = {
                "graph": [obs["graph"] for obs in batch_obs_list],
                "global_state": torch.tensor(np.array([obs["global_state"] for obs in batch_obs_list]), dtype=torch.float32).to(self.device),
                "current_pipe_nodes": torch.tensor(np.array([obs["current_pipe_nodes"] for obs in batch_obs_list]), dtype=torch.long).to(self.device)
            }
            
            yield (
                batch_obs,
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices],
            )
        
    def clear(self):
        """Clear the buffer"""
        self.observations = [None] * self.buffer_size
        self.ptr = 0
        self.size = 0

class GNNPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # --- Extract dimensions from the environment's observation space ---
        # Node features are pressure and demand
        node_features_dim = observation_space['graph'].node_space.shape[0] 
        # Edge features are diameter, length, and a flag for the current pipe
        edge_features_dim = observation_space['graph'].edge_space.shape[0]
        # Global state includes budget, cost, network index, and progress
        global_state_dim = observation_space['global_state'].shape[0]
        
        gnn_out_dim = 128
        combined_features_dim = 256

        # --- GNN Layers ---
        # These layers will process the graph structure of the water network
        self.gnn_layer_1 = GATv2Conv(node_features_dim, gnn_out_dim, edge_dim=edge_features_dim)
        self.gnn_layer_2 = GATv2Conv(gnn_out_dim, gnn_out_dim, edge_dim=edge_features_dim)

        # --- Feature Combiner ---
        # This linear layer will combine features from different sources:
        # 1. The overall graph embedding (from global pooling)
        # 2. The specific embedding of the two nodes connected by the current pipe
        # 3. The global state vector
        combined_input_dim = gnn_out_dim + (gnn_out_dim * 2) + global_state_dim
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_input_dim, combined_features_dim),
            nn.ReLU()
        )

        # --- Actor and Critic Heads ---
        # The actor head decides which action to take
        self.actor = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )
        # The critic head estimates the value of the current state
        self.critic = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _extract_features(self, obs_batch):
        # --- 1. Batch Graph Data ---
        # The input obs_batch["graph"] is a list of GraphInstance objects.
        # We use Batch.from_data_list to combine them into a single, efficient
        # graph object that torch_geometric can process in parallel.
        
        # First, ensure graph data is in the correct PyTorch Geometric Data format
        device = self.gnn_layer_1.lin_l.weight.device
        data_list = []
        for graph_instance in obs_batch["graph"]:
            data_list.append(
                Data(
                    x=torch.tensor(graph_instance.nodes, dtype=torch.float32),
                    edge_index=torch.tensor(graph_instance.edge_links, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(graph_instance.edges, dtype=torch.float32)
                )
            )
        
        batched_graph = Batch.from_data_list(data_list).to(device)

        # --- 2. Process with GNN Layers ---
        # This is the actual Graph Neural Network forward pass.
        # It computes embeddings for every node in the batched graph.
        node_embeddings = self.gnn_layer_1(batched_graph.x, batched_graph.edge_index, edge_attr=batched_graph.edge_attr).relu()
        node_embeddings = self.gnn_layer_2(node_embeddings, batched_graph.edge_index, edge_attr=batched_graph.edge_attr)

        # --- 3. Get Graph-Level Embedding ---
        # We use global mean pooling to get a single vector representing the entire graph.
        # The `batched_graph.batch` tensor tells the pooling function which nodes belong to which graph in the batch.
        graph_embedding = global_mean_pool(node_embeddings, batched_graph.batch)

        # --- 4. Get Specific Node Embeddings for the Current Pipe ---
        # We need to find the embeddings of the start and end nodes of the pipe currently being considered.
        # `obs_batch["current_pipe_nodes"]` holds the indices of these nodes.
        
        # `batched_graph.ptr` helps us map the local node indices from the original graphs
        # to the global indices in the batched graph.
        node_offsets = batched_graph.ptr[:-1]
        absolute_pipe_node_indices = obs_batch["current_pipe_nodes"] + node_offsets.unsqueeze(1)
        
        # Extract the node embeddings for the start and end nodes of the current pipe
        pipe_node_embeddings = node_embeddings[absolute_pipe_node_indices.flatten()].view(len(data_list), -1)

        # --- 5. Combine All Features ---
        # Concatenate the global graph embedding, the specific pipe node embeddings,
        # and the global state vector into a single feature tensor.
        combined_features = torch.cat([
            graph_embedding,
            pipe_node_embeddings,
            obs_batch["global_state"]
        ], dim=1)

        # Pass the combined features through the final processing layer
        return self.feature_combiner(combined_features)

    def get_value(self, x):
        """Returns the state value from the critic network."""
        return self.critic(self._extract_features(x))

    def get_action_and_value(self, x, action=None):
        """
        Returns an action, its log probability, entropy, and the state value.
        """
        hidden = self._extract_features(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":

    args = Args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # Define metrics that use step values
    wandb.define_metric("training/*", step_metric="global_step")
    wandb.define_metric("charts/*", step_metric="global_step")
    wandb.define_metric("losses/*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") # Use GPU if available and specified

    # Env setup
    network_dir = 'Networks/Simple_Nets'
    env = PPOEnv(network_files_dir=network_dir)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = GNNPolicy(env.observation_space, env.action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    wandb.watch(agent, log="all", log_freq=10)

    # Initialize rollout buffer
    buffer_size = args.num_steps
    rb = RolloutBuffer(buffer_size, env.observation_space, env.action_space, device, 
                    gamma=args.gamma, gae_lambda=args.gae_lambda)

    # --- Training Loop ---
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(0, args.total_timesteps, args.num_steps):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (global_step / args.total_timesteps)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # --- COLLECT ROLLOUT ---
        # Clear buffer at the start of each rollout
        
        rb.clear()
        for step in range(args.num_steps):
            current_step = global_step + step
            
            # The new GNNPolicy expects the observation to be batched,
            # so we prepare it as a "batch of 1".
            # obs["graph"] is a GraphInstance object. We put it in a list.
            # obs["global_state"] and obs["current_pipe_nodes"] are NumPy arrays.
            # We convert them to tensors and add a batch dimension.
            agent_obs = {
                "graph": [obs['graph']],
                "global_state": torch.tensor(obs['global_state'], dtype=torch.float32).unsqueeze(0).to(device),
                "current_pipe_nodes": torch.tensor(obs['current_pipe_nodes'], dtype=torch.long).unsqueeze(0).to(device)
            }
            
            # Get action and value
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(agent_obs)
                value = value.flatten()
            
            # Execute action
            # Note: action is a tensor, step() might expect a scalar numpy value
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().item())

            # When adding to the buffer, you add the dictionary directly
            rb.add(
                obs, # Store the original, un-tensored observation
                action.flatten(),
                torch.tensor([reward], device=device),
                torch.tensor([terminated or truncated], device=device),
                value,
                log_prob
            )
            
            # Log if needed (reduce frequency for speed)
            if current_step % 100 == 0:
                print(f"Step: {current_step}")
                wandb.log({
                    "training/action": action.cpu().numpy()[0],
                    "training/reward": reward,
                    "training/value": value.item(),
                    "training/entropy": entropy.item(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"]
                })
            
            # # Add to buffer
            # rb.add(
            #     agent_obs,
            #     action.flatten(),
            #     torch.tensor([reward], device=device),
            #     torch.tensor([terminated or truncated], device=device),
            #     value,
            #     log_prob
            # )
            
            # Handle episode end
            if terminated or truncated:
                if "episode" in info:
                    print(f"global_step={current_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], current_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], current_step)
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                    })
                
                next_obs, _ = env.reset()
            
            obs = next_obs
        
        # --- TRAINING PHASE ---
        # Compute returns and advantages
        with torch.no_grad():
            # Prepare the last observation in the same "batch of 1" format
            last_obs = {
                "graph": [obs['graph']],
                "global_state": torch.tensor(obs['global_state'], dtype=torch.float32).unsqueeze(0).to(device),
                "current_pipe_nodes": torch.tensor(obs['current_pipe_nodes'], dtype=torch.long).unsqueeze(0).to(device)
            }
            last_value = agent.get_value(last_obs).flatten()
        
        # Compute returns and advantages
        rb.compute_returns_and_advantages(last_value)
        
        # PPO Update
        clipfracs = []
        for epoch in range(args.update_epochs):
            # Get batches
            batch_size = args.num_steps // args.num_minibatches
            for batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in rb.get_batches(batch_size):
                # Normalize advantages
                if args.norm_adv:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # Get current action and value
                actions, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
                new_values = new_values.flatten()
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Tracking clipping
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                
                # Clipped surrogate objective
                clip_1 = ratio * batch_advantages
                clip_2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * batch_advantages
                policy_loss = -torch.min(clip_1, clip_2).mean()
                
                # Value loss
                value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
        # Log training metrics
        explained_var = (1 - (rb.returns - rb.values).var() / rb.returns.var()).item()
        
        wandb.log({
            "losses/value_loss": value_loss.item(),
            "losses/policy_loss": policy_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": (batch_old_log_probs - new_log_probs).mean().item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
        })
        
        print(f"Update complete after global step {global_step}")

        # --- Add this line to save the model ---
        model_path = f"runs/{run_name}/agent.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Agent model saved to {model_path}")

        writer.close()
        wandb.finish()