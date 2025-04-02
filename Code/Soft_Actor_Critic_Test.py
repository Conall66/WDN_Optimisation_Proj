
# AI generated SAC RL Model

import numpy as np
import networkx as nx
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define Water Network Environment
class WaterNetworkEnv(gym.Env):
    def __init__(self, num_demand_nodes=5, max_steps=50):
        super(WaterNetworkEnv, self).__init__()
        self.num_demand_nodes = num_demand_nodes
        self.max_steps = max_steps
        self.reset()

        # Continuous action space: [x_offset, y_offset]
        self.action_space = spaces.Box(low=-3, high=3, shape=(2,), dtype=np.float32)
        
        # Observation space: positions of all nodes (flattened)
        self.observation_space = spaces.Box(low=0, high=20, shape=(len(self.graph.nodes) * 2,), dtype=np.float32)
    
    def reset(self):
        self.graph = nx.Graph()
        self.graph.add_node(0, pos=(10, 10))  # Source node
        
        # Generate demand nodes
        self.demand_nodes = {
            i: (np.random.uniform(0, 20), np.random.uniform(0, 20)) 
            for i in range(1, self.num_demand_nodes + 1)
        }
        self.graph.add_nodes_from(self.demand_nodes.items())
        self.steps = 0
        return self._get_observation()
    
    def _get_observation(self):
        positions = nx.get_node_attributes(self.graph, 'pos')
        return np.array(list(positions.values()), dtype=np.float32).flatten()
    
    def step(self, action):
        node_id = random.choice(list(self.graph.nodes))
        x, y = self.graph.nodes[node_id]['pos']
        
        new_pos = (x + action[0], y + action[1])
        new_node_id = len(self.graph.nodes)
        self.graph.add_node(new_node_id, pos=new_pos)
        self.graph.add_edge(node_id, new_node_id, weight=np.linalg.norm(np.array(new_pos) - np.array([x, y])))
        
        # Reward: Minimize total pipe length
        total_length = sum(nx.get_edge_attributes(self.graph, 'weight').values())
        reward = -total_length / 100
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, {}
    
    def render(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue')

# Define SAC Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Ensures continuous values between -1 and 1
        )
    
    def forward(self, x):
        return self.fc(x)

# Training function (Simplified SAC)
def train_sac(env, policy, optimizer, num_episodes=500):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy(state_tensor).detach().numpy().flatten()
            action = action * 3  # Scale action for network expansion
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Simplified SAC Loss (Using Policy Gradient)
            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

# Initialize and train
env = WaterNetworkEnv()
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
optimizer = optim.Adam(policy.parameters(), lr=0.01)
train_sac(env, policy, optimizer, num_episodes=500)

# Test trained model
env.reset()
done = False
while not done:
    obs_tensor = torch.FloatTensor(env._get_observation()).unsqueeze(0)
    action = policy(obs_tensor).detach().numpy().flatten() * 3
    _, _, done, _ = env.step(action)
    env.render()
