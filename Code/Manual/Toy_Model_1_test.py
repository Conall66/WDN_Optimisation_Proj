
# SAC Toy Model 1 - Simple network topology optimisation

# 04/2025 - 06/2025

import numpy as np
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
# from torch import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from scipy.spatial import KDTree
from collections import deque
from multiprocessing import Pool

import os
import matplotlib.pyplot as plt
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU when available

# Input variables
max_nodes = 10 # Maximum number of nodes in the network
max_edges = 30 # Maximum number of edges in the network -- Unused
max_steps = 50 # Maximum number of steps in the environment
max_episodes = 20 # Maximum number of episodes to train the agent
# max_memory = 10000 # Maximum number of experiences to store in memory
batch_size = 128 # Batch size for training
iterations = 5 # Number of iterations of changing demand

grid_size = 20
cost_weights = {
    'reliability': 0.5,
    'length': 50.0, # Experimment with large penalty for increasing length
    'connectivity': 1.0
} # Cost weights for the reward function

# Define actor critic networks for the GNN

"""
Using a GNN rather than feedforward ANN is beneficial here since the environment can be better represented by graphical data. This elimintaed the need to predefine the action space dimensions and waster available memory. The GNN can learn the relationships between nodes and edges in the network, allowing it to make more informed decisions about how to optimise the network.
"""

class GNNActor(nn.Module):
    def __init__(self, node_feat_dim, action_dim, max_action, hidden_dim=256):
        super(GNNActor, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP to produce mean and std from graph embedding
        self.fc = nn.Linear(hidden_dim, action_dim * 2)

        self.max_action = max_action

    def forward(self, x, edge_index, batch):
        # GNN forward pass
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Pool over nodes in each graph to get graph-level embedding
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        mean_log_std = self.fc(x)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, x, edge_index, batch):
        mean, std = self.forward(x, edge_index, batch)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        action_copied = action.clone()
        action_copied[:, 3:] = (action_copied[:, 3:] + 1) / 2 * 20  # maps [-1, 1] to [0, 20]
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action_copied * self.max_action, log_prob
    
class GNNCritic(nn.Module):
    def __init__(self, node_feat_dim, action_dim, hidden_dim=256):
        super(GNNCritic, self).__init__()

        # Two separate Q-networks for Double Q-learning
        self.q1_conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.q1_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.q1_fc = nn.Linear(hidden_dim + action_dim, 1)

        self.q2_conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.q2_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.q2_fc = nn.Linear(hidden_dim + action_dim, 1)

    def forward(self, x, edge_index, batch, action):
        # Q1 stream
        x1 = F.relu(self.q1_conv1(x, edge_index))
        x1 = F.relu(self.q1_conv2(x1, edge_index))
        graph_embed1 = global_mean_pool(x1, batch)
        q1_input = torch.cat([graph_embed1, action], dim=-1)
        q1 = self.q1_fc(q1_input)

        # Q2 stream
        x2 = F.relu(self.q2_conv1(x, edge_index))
        x2 = F.relu(self.q2_conv2(x2, edge_index))
        graph_embed2 = global_mean_pool(x2, batch)
        q2_input = torch.cat([graph_embed2, action], dim=-1)
        q2 = self.q2_fc(q2_input)

        return q1, q2
    
class GraphReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) # Unzip the batch

        # Batch the graphs
        batch_states = Batch.from_data_list(states).to(device)
        batch_next_states = Batch.from_data_list(next_states).to(device)

        # Stack everything else
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        return batch_states, actions, rewards, batch_next_states, dones

    def __len__(self):
        return len(self.buffer)
    
# SAC Agent
class SAC:
    def __init__(
        self, 
        state_dim,
        node_feat_dim, # Number of features for each node in the graph 
        action_dim, 
        max_action, 
        discount=0.99, # Influence of future states
        tau=0.005, # Soft update factor
        alpha=0.2, # Standard val - default is learn alpha = False
        learn_alpha=True, # Change how much estimations affect value funct
        hidden_dim=256 # Standard
    ):
        self.actor = GNNActor(node_feat_dim, action_dim, max_action, hidden_dim).to(device)
        self.critic = GNNCritic(node_feat_dim, action_dim, hidden_dim).to(device)
        self.critic_target = GNNCritic(node_feat_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters from critic network to target critic network and disable gradient for target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        
        # Automatic entropy tuning
        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.target_entropy = -action_dim  # heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimiser = optim.Adam([self.log_alpha], lr=3e-4) # 0.0003
            self.alpha = torch.exp(self.log_alpha)
        else:
            self.alpha = alpha

    def select_action(self, state, edge_index, batch):
        with torch.no_grad():
            state = state
            edge_index = edge_index
            batch = batch
            action, _ = self.actor.sample(state, edge_index, batch)
        return action.cpu().numpy()
    
    def train(self, replay_buffer, batch_size = 256):
        # Sample a batch of experiences from the replay buffer
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

        # Move the batch to the appropriate device (GPU or CPU)
        batch_states = batch_states.to(device)
        batch_next_states = batch_next_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_rewards = batch_rewards.to(device)
        batch_dones = batch_dones.to(device)

        # Compute the target Q-value using the target critic network
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(
                batch_next_states.x.to(device), 
                batch_next_states.edge_index.to(device), 
                batch_next_states.batch.to(device))
            target_q1, target_q2 = self.critic_target(
                batch_next_states.x.to(device), 
                batch_next_states.edge_index.to(device), 
                batch_next_states.batch.to(device), 
                next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.to(device) * next_log_probs.to(device) # From 2 critic networks, take the minimum value
            target_q = batch_rewards + (1 - batch_dones) * self.discount * target_q # Bellman equation

            """
            The Bellman equation describes the relationship between the value of a state and the values of its successor states. It provides a recursive way to compute the value of a state based on the expected rewards and future values of the next states.
            """

        # Critic loss
        q1, q2 = self.critic(
            batch_states.x.to(device),
            batch_states.edge_index.to(device), 
            batch_states.batch.to(device), 
            batch_actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # Mean squared error loss for both Q-values

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # Actor loss
        sample_actions, log_probs = self.actor.sample(
            batch_states.x.to(device), 
            batch_states.edge_index.to(device), 
            batch_states.batch.to(device))
        q1, q2 = self.critic(
            batch_states.x.to(device), 
            batch_states.edge_index.to(device), 
            batch_states.batch.to(device), 
            sample_actions)
        q = torch.min(q1, q2) # Take the minimum Q-value from the two critic networks
        actor_loss = (self.alpha.to(device) * log_probs.to(device) - q).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward() # Undefined?
        self.actor_optimiser.step()

        log_probs =log_probs # Move log_probs to the appropriate device (GPU or CPU)
        # self.target_entropy = self.target_entropy # Move target_entropy to the appropriate device (GPU or CPU)

        # Alpha loss
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha.to(device) * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimiser.zero_grad()
            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = torch.exp(self.log_alpha.to(device))

        # Soft Update of Target Network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Define the environment

class WDNEnvironment:
    def __init__(self, graph, cost_weights, max_nodes = 10): # Takes graph with existing network structure and demand nodes placed

        # Time how long to generate initial graph
        start_time = time.time()

        self.graph = graph
        self.max_nodes = max_nodes # Max nodes represents maximum nodes that can be added at any one iteration in the network
        self.source_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'source']
        self.demand_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'demand']
        self.connector_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'connector']
        self.edges = graph.edges()
        self.total_length = np.sum([np.linalg.norm(np.array(u) - np.array(v)) for u, v in self.edges]) # Calculate the total length of the graph

        self.cost_weights = cost_weights # Cost weights for the reward function

        self.history = {
            'rewards' : [],
            'node_counts' : [],
            'edge_counts' : [],
            'shortest_paths' : [],
            'average_degrees' : [],
            'training_time' : [],
        } # Store values to plot over time

        end_time = time.time()

        # Append initial values to self.history
        self.history['rewards'].append(0)
        self.history['node_counts'].append(len(self.graph.nodes()))
        self.history['edge_counts'].append(len(self.graph.edges()))
        self.history['shortest_paths'].append(0) # Initially there will be disconnected demand nodes
        self.history['average_degrees'].append(np.mean([d for n, d in self.graph.degree()]))
        self.history['training_time'].append(end_time - start_time)

    def reset(self, grid_size = 20): # Reset the environment to its initial state
        self.graph = nx.Graph()
        self.source_nodes = []
        self.demand_nodes = []
        self.connector_nodes = []
        self.edges = [] # Overwritten later
        
        # Add supply node at (0, 0)
        self.graph.add_node((0, 0), type='source') 
        """Currently hard coded! This should be changed to allow for multiple supply nodes in the future"""
        self.source_nodes.append((0, 0))

        # Randomly add connector nodes to generate existing network
        for i in range(random.randint(4, 7)): # Hard coded for now
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='connector')
            self.connector_nodes.append((x, y))

        # Connect initial connector nodes, ensuring at least connection to source nodes
        num_edges = random.randint(len(self.graph.nodes) - 1, self.max_nodes) # Possible to have one fewer connection than no.nodes
        # possible_edges = list(nx.non_edges(graph))
        possible_edges = []

        # For each node, create an array of the 3 closest nodes and randomly add edges connecting them
        for node in self.graph.nodes():
            distances = {other: np.linalg.norm(np.array(node) - np.array(other)) for other in self.graph.nodes() if other != node}
            closest_nodes = sorted(distances, key=distances.get)[:3]  # Get the 3 closest nodes
            for closest_node in closest_nodes:
                possible_edges.append((node, closest_node))

        print(f"Number of possible edges: {len(possible_edges)}")
        print(f"Number of edges to be added: {num_edges}")

        random_edges = random.sample(possible_edges, num_edges)
        self.graph.add_edges_from(random_edges)

        # Ensure every node has at least one connection to another node and connect to the source node
        for node in self.graph.nodes():
            if self.graph.degree(node) == 0:
                # Find the nearest node
                nearest_node = min(self.graph.nodes(), key=lambda n: np.linalg.norm(np.array(node) - np.array(n)))
                # Connect the node to the nearest node
                self.graph.add_edge(node, nearest_node)
        
        for node in self.graph.nodes():
            # Check every node has a path to the source node
            if not nx.has_path(self.graph, (0, 0), node):
                self.graph.add_edge((0, 0), node)

        # Randomly generate demand nodes
        # Calculate number of existing nodes and subtract from max_nodes
        num_demand_nodes = random.randint(3, self.max_nodes - len(self.connector_nodes))
        for i in range(num_demand_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='demand')
            self.demand_nodes.append((x, y))

        return self.get_state()
    
    def upd_state(self, grid_size=20): # Update the state of the environment for next iteration by adding new demand nodes

        """Currently this function is not being called. The agent must properly learn to design for a single optimised network before being able to adapt to changing demand nodes over time"""

        for i in range(self.max_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='demand')
            self.demand_nodes.append((x, y))

        return self.get_state()

    def to_pyg_data(self):
        # Convert networkx graph to PyTorch Geometric Data
        node_features = []
        node_pos = []
        type_map = {'source': 0, 'connector': 1, 'demand': 2}
        mapping = {n: i for i, n in enumerate(self.graph.nodes())}

        for node in self.graph.nodes(data=True):
            pos = np.array(node[0])
            typ = type_map[node[1]['type']]
            node_features.append([*pos, typ])
            node_pos.append(node[0])

        # print(f"Length of networkx edges: {len(self.graph.edges)}")    
        edge_index = torch.tensor([[mapping[u], mapping[v]] for u, v in self.graph.edges()], dtype=torch.long).t().contiguous().to(device)
        # print(f"Length of edge_index: {len(edge_index)}")
        x = torch.tensor(node_features, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index).to(device)
    
    def step(self, action): # Take a step in the environment with the given action

        # print("New Step")
        start_time = time.time()

        # Action in form (action_type(3), x, y) where action_type is 0 (no action), 1 (add node), 2 (connect nodes)
        action_type = np.argmax(action[:3]) # The first 3 elements of the action functions are the probabilities of a reward given each possible action

        if action_type == 0: # Gives the agent the chance to skip any action if no reward is better than alternative solutions 
            print("No action taken")
            pass # No action taken
        elif action_type == 1: # Add new node in specified position
            print("Adding new node")
            x = int(round(action[3])) # Extract suggested position values from action function
            y = int(round(action[4]))
            proposed_position = (x, y)
            if proposed_position not in self.graph.nodes(): # and len(self.graph.nodes()) < self.max_nodes:
                # Add a new node to the graph
                self.graph.add_node(proposed_position, type='connector')
                self.connector_nodes.append(proposed_position)

                # Connect the new node to the nearest existing node excluding itself
                nearest_node = min(
                    (n for n in self.graph.nodes() if n != proposed_position),  # Exclude the node itself
                    key=lambda n: np.linalg.norm(np.array(proposed_position) - np.array(n))
                )
                self.graph.add_edge(proposed_position, nearest_node)
                self.edges = self.graph.edges()
                self.total_length += np.linalg.norm(np.array(proposed_position) - np.array(nearest_node)) # Update the total length of the graph
            else:
                print("Proposed position already exists or max nodes reached") # Debugging line

        elif action_type == 2: # Further connect existing nodes
            print("Connecting existing nodes")
            # Identify the least connected node in the graph
            least_connected_node = min(self.graph.nodes(), key=lambda n: self.graph.degree(n))
            # Rank by order of distance - node 1 should be the least connected node itself
            reordered_nodes = sorted(self.graph.nodes(), key=lambda n: np.linalg.norm(np.array(least_connected_node) - np.array(n)))
            for i in range(1, len(reordered_nodes)):
                # Connect the least connected node to the next node in the list
                if reordered_nodes[i] != least_connected_node:
                    # Check if the edge already exists
                    if not self.graph.has_edge(least_connected_node, reordered_nodes[i]):
                        self.graph.add_edge(least_connected_node, reordered_nodes[i])
                        self.edges = self.graph.edges()
                        self.total_length += np.linalg.norm(np.array(least_connected_node) - np.array(reordered_nodes[i])) # Update the total length of the graph
                        break   

        # connected = all(nx.has_path(self.graph, source, target) 
        #                  for source in self.demand_nodes 
        #                  for target in self.demand_nodes 
        #                  if source != target)

        demand_components = set()
        for component in nx.connected_components(self.graph):
            demand_components.add(frozenset(component))

        connected = all(any(node in component for component in demand_components) for node in self.demand_nodes)

        # visualise_network(self.graph) # Visualise the graph after each step for debugging
        
        reliability = self.calculate_reliability()

        # print(f"Reliability: {reliability}")
        # print(f"Length: {self.total_length}")

        # Calculate the reward values
        w1, w2, w3 = self.cost_weights['reliability'], self.cost_weights['length'], self.cost_weights['connectivity'] # Unpack the cost weights
        reward = w1 * reliability - w2 * (self.total_length/1000) # Calculate the reward based on the cost weights (reward reliability whilst punishing unnecessary length)

        # Heavy penalty for disconnected demand nodes
        if not connected:
            reward -= w3

        # Get new state
        next_state = self.get_state()
        done = not connected # End the episode if there are disconnected nodes (reward has heavy low value)
        info = {
            'reliability': reliability,
            'length': self.total_length,
            'connected': connected
        }

        end_time = time.time()

        self.history['rewards'].append(reward)
        self.history['node_counts'].append(len(self.graph.nodes()))
        self.history['edge_counts'].append(len(self.graph.edges()))
        # if not connected:
        if nx.is_connected(self.graph):
            self.history['shortest_paths'].append(nx.average_shortest_path_length(self.graph)) # In first few iterations model does not yet know to connect all nodes
        else:
            self.history['shortest_paths'].append(int(0))
        self.history['average_degrees'].append(np.mean([d for n, d in self.graph.degree()]))
        self.history['training_time'].append(end_time - start_time)

        return next_state, reward, done, info # Return the new state, reward, done (False), and additional information

    def get_state(self):
        # Returns the current state of the environment
        # State can be represented as a list of node positions and their types
        state = []
        # Inidvidual features
        for node in self.graph.nodes(data=True):
            position, node_type = node
            state.append((position, node_type['type']))
        # Global features
        state.append(len(self.graph.nodes()))
        state.append(len(self.graph.edges()))
        state.append(self.calculate_reliability())
        state.append(self.calculate_length())

        return state # Returns state in the form (node_idx, position, type) * num_nodes + (num_nodes, num_edges, reliability, length)
    
    def calculate_reliability(self):
        return nx.average_node_connectivity(self.graph)
    
    def calculate_length(self):
        # Calculate the total length of the edges in the graph
        return self.total_length
    
    # Returns error if new nodes are added and are not connected to the graph
    
# ____ Train Water Network Function ______________________________________________________________________

def optimised_wdn(env, cost_weights, max_nodes):

    # env = WDNEnvironment(graph_dimensions, initial_nodes, initial_edges, quant_demand_nodes)
    env = WDNEnvironment(env, cost_weights, max_nodes) # Cost weights for the reward function

    # Initialise the replay buffer
    replay_buffer = GraphReplayBuffer()

    # Initialise the agent
    state_dim = len(env.get_state()) # Number of features in the state space
    node_feat_dim = 3 # Number of features for each node in the graph (position and type)
    action_dim = 5 # Number of actions available to the agent (add node, connect nodes, remove node)
    max_action = 1.0 # Maximum action value (for continuous action space)
    
    agent = SAC(state_dim, node_feat_dim, action_dim, max_action, hidden_dim = 256)

    # Training loop
    for episode in range(max_episodes):
        env.reset(grid_size) # Reset the environment at the start of each episode
        done = False
        step_count = 0

        print(f"Episode {episode + 1}/{max_episodes}")
        print("--------------------------------------------------")
        
        while not done and step_count < max_steps:
            pyg_state = env.to_pyg_data() # Converts networkx graph to PyTorch Geometric Data format
            # print(pyg_state) # Data(x = [10,3], edge_index = [0])
            batch = Batch.from_data_list([pyg_state])
            action = agent.select_action(batch.x, batch.edge_index, batch.batch)
            action_np = action.flatten() # Convert action to numpy array and flatten it
            # print(f"Action_np: {action_np}") # Print the selected action
            # print(f"Action shape: {action_np.shape}") # Print the shape of the action

            next_state, reward, done, info = env.step(action_np) # Take a step in the environment with the selected action
            next_pyg_state = env.to_pyg_data() # Convert the next state to PyTorch Geometric Data format
            replay_buffer.add(pyg_state, torch.tensor(action_np, dtype=torch.float32), reward, next_pyg_state, done)

            pyg_state = next_pyg_state
            step_count += 1

            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size) # Train the agent using experiences from the replay buffer

        visualise_network(env.graph) # Visualise the graph after each step for debugging

        # Check all nodes are connected
        
        """This is going to drastically slow down model - connection should be checked every time new node generated instead"""

        print(f"Episode {episode + 1}/{max_episodes} completed.")
        print(f"Total reward: {sum(env.history['rewards'])}")
        print(f"Number of nodes: {len(env.graph.nodes())}")
        print(f"Number of edges: {len(env.graph.edges())}")
        print(f"Average degree: {np.mean([d for n, d in env.graph.degree()])}")
        # if connected:
        #     print(f"Average shortest path length: {nx.average_shortest_path_length(env.graph)}")
        # else:
        # print("Graph is disconnected.")
        print(f"Reliability: {env.calculate_reliability()}")
        print(f"Length: {env.calculate_length()}")
        print("--------------------------------------------------")

    visualise_network(env.graph) # Visualise the final graph after training

    visualise_training(env.history) # Visualise the training history

    print("Training history: ", env.history) # Print the training history

def main(max_nodes, cost_weights, grid_size):

    print(f"Using device: {device}")
    print(f"GPU Available: {torch.cuda.is_available()}")

    # Generate initial environment to pass to optimised_wdn()
    graph = nx.Graph()

    # Add source node
    graph.add_node((0, 0), type='source')

    # Generate intial connector nodes
    for i in range(random.randint(4, 6)): # Hard coded for now
        x = random.randint(0, grid_size)
        y = random.randint(0, grid_size)
        graph.add_node((x, y), type='connector')

    # Connect initial connector nodes, ensuring at least connection to source nodes
    num_edges = random.randint(len(graph.nodes) - 1, len(graph.nodes) + 4) # Possible to have one fewer connection than no.nodes
    # possible_edges = list(nx.non_edges(graph))
    possible_edges = []

    # For each node, create an array of the 3 closest nodes and randomly add edges connecting them
    for node in graph.nodes():
        distances = {other: np.linalg.norm(np.array(node) - np.array(other)) for other in graph.nodes() if other != node} # Find distance to all other nodes
        closest_nodes = sorted(distances, key=distances.get)[:4]  # Get the 3 closest nodes + node itself
        for closest_node in closest_nodes:
            if node != closest_node:
                possible_edges.append((node, closest_node)) # Each node should add at least 3 edges as long as the number of initial nodes is equal to or greater than 4

    # print(f"Possible edges: {len(possible_edges)}")
    # print(f"Number of edges to be added: {num_edges}")

    # In some instances, the number of possible edges seems to be smaller than the number of edges ot be added...

    random_edges = random.sample(possible_edges, num_edges) # samples random edges from the possible edges
    graph.add_edges_from(random_edges)

    # Ensure every node has at least one connection to another node and the source nodes
    for node in graph.nodes():
        if graph.degree(node) == 0:
            # Find the nearest node
            nearest_node = min(graph.nodes(), key=lambda n: np.linalg.norm(np.array(node) - np.array(n)))
            # Connect the node to the nearest node
            graph.add_edge(node, nearest_node)
    
    for node in graph.nodes():
        # Check every node has a path to the source node
        if not nx.has_path(graph, (0, 0), node):
            graph.add_edge((0, 0), node)

    # Generate initial demand nodes
    for i in range(random.randint(3, (max_nodes - len(graph.nodes())))): # Want at least a few demand nodes
        x = random.randint(0, grid_size)
        y = random.randint(0, grid_size)
        graph.add_node((x, y), type='demand')

    # Visualise initial graph
    visualise_network(graph)
    
    # Pass to optimised_wdn() function to see training results
    optimised_wdn(graph, cost_weights, max_nodes)

def visualise_network(graph):
    pos = {node: node for node in graph.nodes()}
    node_types = nx.get_node_attributes(graph, 'type')
    node_colors = {'source': 'blue', 'connector': 'green', 'demand': 'red'}
    colors = [node_colors[node_types[node]] for node in graph.nodes()]
    nx.draw(graph, pos, with_labels=True, node_color=colors, node_size=500, font_size=8)
    plt.title("Optimised Graph")
    plt.show(block = False)
    plt.pause(5)
    plt.close()

def visualise_training(history): # Plot how reward function updates with each episode

    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(history['rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward over Episodes')

    # Plot training time and number of nodes
    plt.subplot(1, 2, 2)

    # Plot training time on the left y-axis
    ax1 = plt.gca()  # Get the current axis
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Training Time (seconds)', color='blue')
    ax1.plot(history['training_time'], label='Training Time', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for the number of nodes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Nodes', color='orange')
    ax2.plot(history['node_counts'], label='Number of Nodes', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add a title
    plt.title('Training Time and Number of Nodes over Episodes')

    plt.tight_layout()
    plt.show()

main(max_nodes, cost_weights, grid_size)
# End of code