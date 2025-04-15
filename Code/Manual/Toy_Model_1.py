
# SAC Toy Model 1 - Simple network topology optimisation

# 11/04/2025

"""
Methodology:

Action Space:
- Generate new nodes
- Connect existing nodes
- Remove existing nodes

Observation Space:
- Nodes generated
- Positions of new nodes
- Connections between nodes (degree)
- Node status/types (source, connector, demand)

Rewards/Penalties:
- Minimised length: -0.5 * length of edges
- Connectivity to demand nodes: +1 for each demand node connected, -1 for each disconnected node
- Maximised connectivity: +0.5 for each node degree above 1

"""

# Generate a training environment

# Generate a random number of nodes between 1 and 10
# 0.5 Probability to connect to nodes within 3 units proximity

import numpy as np
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from collections import deque
import random

# Input variables
max_nodes = 10 # Maximum number of nodes in the network
max_edges = 30 # Maximum number of edges in the network
max_steps = 100 # Maximum number of steps in the environment
max_episodes = 100 # Maximum number of episodes to train the agent
# max_memory = 10000 # Maximum number of experiences to store in memory
batch_size = 256 # Batch size for training
iterations = 5 # Number of iterations of changing demand

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
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob
    
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
        self.actor = GNNActor(state_dim, node_feat_dim, action_dim, max_action, hidden_dim)
        self.critic = GNNCritic(state_dim, node_feat_dim, action_dim, hidden_dim)
        self.critic_target = GNNCritic(state_dim, action_dim, hidden_dim)
        
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

# Define the environment

class WDNEnvironment:
    def __init__(self, graph, cost_weights, max_nodes = 10): # Takes graph with existing network structure and demand nodes placed
        self.graph = graph
        self.max_nodes = max_nodes # Max nodes represents maximum nodes that can be added at any one iteration in the network
        self.source_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'source']
        self.demand_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'demand']
        self.connector_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'connector']
        self.edges = graph.edges()

        self.cost_weights = cost_weights # Cost weights for the reward function

        self.history = {
            'rewards' : [],
            'node_counts' : [],
            'edge_counts' : [],
            'shortest_paths' : [],
            'average_degrees' : []
        } # Store values to plot over time

    def reset(self, grid_size = 20): # Reset the environment to its initial state
        self.graph = nx.Graph()
        self.source_nodes = []
        self.demand_nodes = []
        self.connector_nodes = []
        self.edges = [] # Overwritten later
        
        # Add supply node at (0, 0)
        self.graph.add_node((0, 0), type='source')
        self.source_nodes.append((0, 0))

        # Randomly add connector nodes to generate existing network
        for i in range(random.randint(1, self.max_nodes)):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='connector')
            self.connector_nodes.append((x, y))

        # Generate a set of initial edges that is at least len(nodes) - 1
        num_edges = random.randint(len(self.connector_nodes) - 1, len(self.connector_nodes) + 4) # Arbitrary top end
        potential_edges = list(nx.non_edges(self.graph))
        random_edges = random.sample(potential_edges, num_edges)
        self.graph.add_edges_from(random_edges)
        self.edges = self.graph.edges()

        # Randomly generate demand nodes
        # Calculate number of existing nodes and subtract from max_nodes
        num_demand_nodes = random.randint(1, self.max_nodes - len(self.connector_nodes))
        for i in range(num_demand_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='demand')
            self.demand_nodes.append((x, y))

        return self.get_state()
    
    def upd_state(self, grid_size=20): # Update the state of the environment for next iteration by adding new demand nodes

        for i in range(self.max_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            self.graph.add_node((x, y), type='demand')
            self.demand_nodes.append((x, y))

        return self.get_state()

    def step(self, action):

        # Action in form (action_type(3), x, y) where action_type is 0 (no action), 1 (add node), 2 (connect nodes)
        action_type = np.argmax(action[:3]) # The first 3 elements of the action functions are the probabilities of a reward given each possible action

        if action_type == 0: # Gives the agent the chance to skip any action if no reward is better than alternative solutions 
            pass # No action taken
        elif action_type == 1: # Add new node in specified position
            x = action[3] # Extract suggested position values from action function
            y = action[4]
            proposed_position = (x, y)
            if proposed_position not in self.graph.nodes() and len(self.graph.nodes()) < self.max_nodes:
                # Add a new node to the graph
                self.graph.add_node(proposed_position, type='connector')
                self.connector_nodes.append(proposed_position)

                # Connect the new node to the nearest existing node
                nearest_node = min(self.graph.nodes(), key=lambda n: np.linalg.norm(np.array(proposed_position) - np.array(n)))
                self.graph.add_edge(proposed_position, nearest_node)
                self.edges = self.graph.edges()
        elif action_type == 2: # Further connect existing nodes
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
                        break   

        connected = all(nx.has_path(self.graph, source, target) 
                         for source in self.demand_nodes 
                         for target in self.demand_nodes 
                         if source != target)
        
        reliability = self.calculate_reliability()
        length = self.calculate_length()

        # Calculate the reward values
        w1, w2, w3 = self.cost_weights['reliability'], self.cost_weights['length'], self.cost_weights['connectivity'] # Unpack the cost weights
        reward = w1 * reliability - w2 * length # Calculate the reward based on the cost weights (reward reliability whilst punishing unnecessary length)

        # Heavy penalty for disconnected demand nodes
        if not connected:
            reward -= w3

        # Get new state
        next_state = self.get_state()
        done = not connected # End the episode if there are disconnected nodes (reward has heavy low value)
        info = {
            'reliability': reliability,
            'length': length,
            'connected': connected
        }

        self.history['rewards'].append(reward)
        self.history['node_counts'].append(len(self.graph.nodes()))
        self.history['edge_counts'].append(len(self.graph.edges()))
        self.history['shortest_paths'].append(nx.average_shortest_path_length(self.graph))
        self.history['average_degrees'].append(np.mean([d for n, d in self.graph.degree()]))

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
        return nx.average_shortest_path_length(self.graph)
    