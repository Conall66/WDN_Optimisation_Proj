import numpy as np
import networkx as nx
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim

# Define Water Distribution Network Environment
class WaterNetworkEnv(gym.Env):
    def __init__(self, num_demand_nodes=5, max_steps=50):
        super(WaterNetworkEnv, self).__init__()
        self.num_demand_nodes = num_demand_nodes
        self.max_steps = max_steps
        self.reset()

        # Action space: choosing a node to expand & direction
        self.action_space = spaces.Discrete(len(self.graph.nodes) * 4)  # 4 directions per node
        
        # Observation space: adjacency matrix + node features (normalized positions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.graph.nodes), 2), dtype=np.float32)
    
    def reset(self):
        self.graph = nx.Graph()
        self.graph.add_node(0, pos=(0, 0))  # Source node
        
        # Generate demand nodes
        self.demand_nodes = {
            i: (np.random.uniform(5, 20), np.random.uniform(5, 20)) 
            for i in range(1, self.num_demand_nodes + 1)
        }
        self.graph.add_nodes_from(self.demand_nodes.items())
        self.steps = 0
        return self._get_observation()
    
    def _get_observation(self):
        positions = nx.get_node_attributes(self.graph, 'pos')
        return np.array(list(positions.values()), dtype=np.float32)
    
    def step(self, action):
        node_id = action // 4  # Select node
        direction = action % 4  # Choose expansion direction
        
        if node_id not in self.graph.nodes:
            return self._get_observation(), -1, False, {}
        
        x, y = self.graph.nodes[node_id]['pos']
        
        if direction == 0:
            new_pos = (x + np.random.uniform(1, 3), y)  # Right
        elif direction == 1:
            new_pos = (x - np.random.uniform(1, 3), y)  # Left
        elif direction == 2:
            new_pos = (x, y + np.random.uniform(1, 3))  # Up
        else:
            new_pos = (x, y - np.random.uniform(1, 3))  # Down
        
        new_node_id = len(self.graph.nodes)
        self.graph.add_node(new_node_id, pos=new_pos)
        self.graph.add_edge(node_id, new_node_id, weight=np.linalg.norm(np.array(new_pos) - np.array([x, y])))
        
        # Reward: Minimize total pipe length while expanding
        total_length = sum(nx.get_edge_attributes(self.graph, 'weight').values())
        reward = -total_length / 100  # Penalize longer networks
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, {}
    
    def render(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue')

# Define a simple Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Train using Reinforce Algorithm
def train_reinforce(env, policy, optimizer, num_episodes=500):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
            action_probs = policy(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
            
            state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
        
        # Compute loss
        discounted_rewards = np.array([sum(rewards[i:] * (0.99 ** i) for i in range(len(rewards)))])
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        loss = -sum(log_probs) * discounted_rewards
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

# Initialize and train policy
env = WaterNetworkEnv()
policy = PolicyNetwork(env.observation_space.shape[0] * 2, env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
train_reinforce(env, policy, optimizer, num_episodes=500)

# Test the trained model
obs = env.reset()
done = False
while not done:
    obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)
    action_probs = policy(obs_tensor)
    action = torch.multinomial(action_probs, 1).item()
    obs, reward, done, _ = env.step(action)
    env.render()
