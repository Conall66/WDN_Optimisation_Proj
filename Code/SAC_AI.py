
# Claude generated SAC Test

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time

# Neural network for the Actor (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_log_std = self.l3(a)
        
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2) # Keeps sd in reasonable range
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterisation trick enables backprop
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

# Neural network for the Critic (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Double Q Learning reduces overestimation bias

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for stability)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity) # sets maximum number of trials to store in replay buffer at 1000000
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# SAC Agent
class SAC:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action, 
        discount=0.99, # Influence of future states
        tau=0.005, # Soft update factor
        alpha=0.2, # Standard val - default is learn alpha = False
        learn_alpha=True, # Change how much estimations affect value funct
        hidden_dim=256 # Standard
    ):
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        
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
    
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if evaluate:
                mean, _ = self.actor(state)
                return torch.tanh(mean) * self.max_action
            else:
                action, _ = self.actor.sample(state)
                return action.cpu().numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        # Sample a batch from memory
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = reward + (1 - done) * self.discount * target_q
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()
        
        # Update actor
        action_new, log_pi = self.actor.sample(state)
        q1, q2 = self.critic(state, action_new)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_pi - q).mean()
        
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()
        
        # Update alpha (automatic entropy tuning)
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimiser.zero_grad()
            alpha_loss.backward()
            self.alpha_optimiser.step()
            
            self.alpha = torch.exp(self.log_alpha)
        
        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if self.learn_alpha else self.alpha
        }

# Water Distribution Network Environment (basic skeleton)
class WaterNetworkEnv:
    def __init__(self, initial_graph, node_demands, cost_weights):
        self.G = initial_graph.copy()
        self.node_demands = node_demands  # Dict mapping nodes to demand values
        self.cost_weights = cost_weights  # Weights for different cost components
        self.demand_nodes = set(node_demands.keys())
        self.connector_nodes = set(self.G.nodes()) - self.demand_nodes # Connector nodes are all nodes - demand nodes
        
        # For visualization and tracking
        self.history = {
            'rewards': [],
            'entropies': [],
            'costs': [],
            'reliabilities': [],
            'node_counts': [],
            'edge_counts': [],
            'total_pipe_length': []
        }
        
    def calculate_entropy(self):
        # This is a placeholder for your entropy calculation
        # Possible implementations include:
        # 1. Flow diversity (how many different paths water can take)
        # 2. Redundancy measures
        # 3. Resilience to node/edge failures
        
        # Example: simple path diversity metric
        entropy = 0
        for source in self.demand_nodes:
            for target in self.demand_nodes:
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(self.G, source, target, cutoff=10))
                        if paths:
                            entropy += np.log(len(paths))
                    except:
                        pass
        return entropy
    
    def calculate_cost(self):
        # Calculate total network cost based on:
        # 1. Pipe lengths
        # 2. Pipe diameters
        # 3. Number of connections
        # 4. Pumping costs, etc.
        
        total_cost = 0
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 1.0)
            diameter = data.get('diameter', 1.0)
            # Cost model: length * diameter^2 (simplified)
            total_cost += length * (diameter ** 2)
            
        # Add fixed costs for nodes
        for node in self.G.nodes():
            if node in self.connector_nodes:
                total_cost += self.cost_weights.get('connector_cost', 5.0)
                
        return total_cost
    
    def calculate_reliability(self):
        # Calculate network reliability
        # This could be based on:
        # 1. Minimum cut size
        # 2. Node connectivity
        # 3. Edge connectivity
        # 4. Flow satisfaction under failure scenarios
        
        # Example: using average node connectivity
        return nx.average_node_connectivity(self.G)

    def get_state(self):
        # Convert relevant network properties to a state vector
        # This is highly problem-specific and would need customization
        
        # Basic example:
        state = []
        
        # Node features
        for node in sorted(self.G.nodes()):
            # Node position (if available)
            pos = self.G.nodes[node].get('pos', (0, 0))
            state.extend(pos)
            
            # Node demand (0 for connector nodes)
            demand = self.node_demands.get(node, 0)
            state.append(demand)
            
            # Node type (1 for demand, 0 for connector)
            node_type = 1 if node in self.demand_nodes else 0
            state.append(node_type)
        
        # Global features
        state.append(self.calculate_entropy())
        state.append(self.calculate_cost())
        state.append(self.calculate_reliability())
        
        return np.array(state, dtype=np.float32)
    
    def calculate_total_pipe_length(self):
        """Calculate the total length of all pipes in the network"""
        total_length = 0
        for u, v, data in self.G.edges(data=True):
            total_length += data.get('length', 1.0)
        return total_length
    
    def step(self, action):
        # Interpret and apply the action to modify the network
        # Actions could include:
        # 1. Add connector node
        # 2. Remove connector node
        # 3. Add edge
        # 4. Remove edge
        # 5. Modify pipe diameter
        
        # This is a simplified example - your actual implementation will be more complex
        action_type = np.argmax(action[:3])  # First 3 elements determine action type
        
        if action_type == 0:  # Add connector node
            x, y = action[3:5]  # Node position
            new_node = f"c{len(self.connector_nodes)}"
            self.G.add_node(new_node, pos=(x, y))
            self.connector_nodes.add(new_node)
            
            # Connect to closest nodes (up to 3)
            nodes = list(self.G.nodes())
            nodes.remove(new_node)
            distances = []
            
            for node in nodes:
                node_pos = self.G.nodes[node].get('pos', (0, 0))
                dist = np.sqrt((node_pos[0] - x)**2 + (node_pos[1] - y)**2) # pythagoras
                distances.append((node, dist))
            
            # Connect to closest nodes
            closest_nodes = sorted(distances, key=lambda x: x[1])[:3]
            for node, dist in closest_nodes:
                diameter = max(0.1, action[5])  # Diameter from action
                self.G.add_edge(new_node, node, length=dist, diameter=diameter)
                
        elif action_type == 1:  # Remove connector node (if safe)
            if len(self.connector_nodes) > 0:
                # Select connector node based on action
                idx = int(action[3] * len(self.connector_nodes)) % len(self.connector_nodes)
                node_to_remove = list(self.connector_nodes)[idx]
                
                # Check if removal would disconnect the graph
                temp_G = self.G.copy()
                temp_G.remove_node(node_to_remove)
                
                # Only remove if all demand nodes are still connected
                if all(nx.has_path(temp_G, source, target) 
                       for source in self.demand_nodes 
                       for target in self.demand_nodes 
                       if source != target):
                    self.G.remove_node(node_to_remove)
                    self.connector_nodes.remove(node_to_remove)
                
        elif action_type == 2:  # Modify pipe diameters
            for u, v in self.G.edges():
                # Get existing diameter
                current_diameter = self.G[u][v].get('diameter', 1.0)
                
                # Scale by factor from action (bounded)
                scale_factor = max(0.5, min(1.5, action[6]))
                new_diameter = current_diameter * scale_factor
                
                # Update diameter
                self.G[u][v]['diameter'] = new_diameter

        # Calculate reward components
        entropy = self.calculate_entropy()
        cost = self.calculate_cost()
        reliability = self.calculate_reliability()
        total_pipe_length = self.calculate_total_pipe_length()
        
        # Check if all demand nodes are connected
        connected = all(nx.has_path(self.G, source, target) 
                         for source in self.demand_nodes 
                         for target in self.demand_nodes 
                         if source != target)
        
        # Reward function
        w1, w2, w3, w4 = self.cost_weights['entropy'], self.cost_weights['cost'], \
                          self.cost_weights['reliability'], self.cost_weights['connectivity']
        
        reward = w1 * entropy - w2 * cost + w3 * reliability
        
        # Heavy penalty for disconnected demand nodes
        if not connected:
            reward -= w4
        
        # Get new state
        next_state = self.get_state()
        
        # Check if episode should terminate
        done = not connected or cost > self.cost_weights['max_cost']
        
        info = {
            'entropy': entropy,
            'cost': cost,
            'reliability': reliability,
            'connected': connected
        }

        # Track metrics for visualization
        self.history['entropies'].append(entropy)
        self.history['costs'].append(cost)
        self.history['reliabilities'].append(reliability)
        self.history['node_counts'].append(len(self.G.nodes()))
        self.history['edge_counts'].append(len(self.G.edges()))
        self.history['total_pipe_length'].append(total_pipe_length)
        self.history['rewards'].append(reward)
        
        return next_state, reward, done, info
    
    def reset(self):
        # Reset to initial configuration or generate a new starting point
        self.G = nx.Graph()
        
        # Add demand nodes at random positions
        for i, node in enumerate(self.node_demands.keys()):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            self.G.add_node(node, pos=(x, y))
        
        # Connect demand nodes to form an initial connected graph
        for i, node1 in enumerate(self.node_demands.keys()):
            pos1 = self.G.nodes[node1]['pos']
            for j, node2 in enumerate(self.node_demands.keys()):
                if i < j:  # Connect each pair only once
                    pos2 = self.G.nodes[node2]['pos']
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    self.G.add_edge(node1, node2, length=dist, diameter=1.0)
        
        # Reset connector nodes
        self.connector_nodes = set()

        self.history = {
            'rewards': [],
            'entropies': [],
            'costs': [],
            'reliabilities': [],
            'node_counts': [],
            'edge_counts': [],
            'total_pipe_length': []
        }
        
        return self.get_state()
    
    def visualize_network(self, episode=None, save_path=None):
        """
        Visualize the current network topology with node types and pipe diameters
        """
        plt.figure(figsize=(10, 8))
        
        # Create position mapping for all nodes
        pos = nx.get_node_attributes(self.G, 'pos')
        if not pos:  # If positions are not defined, use spring layout (flexible)
            pos = nx.spring_layout(self.G)
        
        # Draw demand nodes (red)
        nx.draw_networkx_nodes(
            self.G, pos, 
            nodelist=list(self.demand_nodes),
            node_color='red',
            node_size=300,
            alpha=0.8
        )
        
        # Draw connector nodes (blue)
        nx.draw_networkx_nodes(
            self.G, pos, 
            nodelist=list(self.connector_nodes),
            node_color='blue',
            node_size=200,
            alpha=0.6
        )
        
        # Calculate edge widths based on pipe diameters
        edge_widths = []
        for u, v in self.G.edges():
            diameter = self.G[u][v].get('diameter', 1.0)
            edge_widths.append(diameter * 2)  # Scale for visualization
        
        # Draw edges with varying widths
        nx.draw_networkx_edges(
            self.G, pos,
            width=edge_widths,
            alpha=0.7,
            edge_color='gray'
        )
        
        # Add labels to demand nodes only for clarity
        demand_labels = {node: f"{node}\n({self.node_demands[node]})" for node in self.demand_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=demand_labels, font_size=10)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Demand Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Connector Node'),
            Line2D([0], [0], color='gray', lw=1, label='Pipe (Thin)'),
            Line2D([0], [0], color='gray', lw=4, label='Pipe (Thick)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add stats as text
        stats_text = (
            f"Nodes: {len(self.G.nodes())} ({len(self.demand_nodes)} demand, {len(self.connector_nodes)} connector)\n"
            f"Edges: {len(self.G.edges())}\n"
            f"Cost: {self.calculate_cost():.2f}\n"
            f"Entropy: {self.calculate_entropy():.2f}\n"
            f"Reliability: {self.calculate_reliability():.2f}\n"
            f"Total Pipe Length: {self.calculate_total_pipe_length():.2f}"
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f"Water Distribution Network{' - Episode ' + str(episode) if episode is not None else ''}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_learning_curves(self, save_path=None, smoothing=10):
        """
        Plot the learning curves for various metrics
        """
        if not self.history['rewards']:
            print("No history to plot yet.")
            return
        
        # Apply smoothing
        def smooth(data, window=smoothing):
            if len(data) < window:
                return data
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                smoothed.append(sum(data[start:end]) / (end - start))
            return smoothed
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot rewards
        axs[0].plot(smooth(self.history['rewards']), 'b-', label='Reward')
        axs[0].set_title('Training Progress')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot cost and entropy
        ax2 = axs[1]
        ax2.plot(smooth(self.history['costs']), 'r-', label='Cost')
        ax2.set_ylabel('Cost', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(smooth(self.history['entropies']), 'g-', label='Entropy')
        ax2_twin.set_ylabel('Entropy', color='g')
        ax2_twin.tick_params(axis='y', labelcolor='g')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Plot reliability and pipe length
        ax3 = axs[2]
        ax3.plot(smooth(self.history['reliabilities']), 'c-', label='Reliability')
        ax3.set_ylabel('Reliability', color='c')
        ax3.tick_params(axis='y', labelcolor='c')
        ax3.grid(True)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(smooth(self.history['total_pipe_length']), 'm-', label='Pipe Length')
        ax3_twin.set_ylabel('Pipe Length', color='m')
        ax3_twin.tick_params(axis='y', labelcolor='m')
        
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='best')
        
        ax3.set_xlabel('Steps')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# Example usage
def train_water_network_agent(env, num_episodes=1000, batch_size=256, max_steps=100, 
                              visualize_interval=50, output_dir='results'):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    state_dim = len(env.get_state())
    action_dim = 7  # Example: [action_type(3), x, y, node_idx, diameter, scale_factor]
    max_action = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 2.0, 1.5])
    
    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    # For tracking overall progress
    all_episode_rewards = []
    all_final_metrics = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update state and cumulative reward
            state = next_state
            episode_reward += reward
            
            # Train agent
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size)
            
            if done:
                break
        
        all_episode_rewards.append(episode_reward)
        
        # Store final metrics for this episode
        final_metrics = {
            'episode': episode,
            'reward': episode_reward,
            'steps': step+1,
            'entropy': info['entropy'],
            'cost': info['cost'],
            'reliability': info['reliability'],
            'nodes': len(env.G.nodes()),
            'edges': len(env.G.edges()),
            'pipe_length': env.calculate_total_pipe_length()
        }
        all_final_metrics.append(final_metrics)
        
        # Print episode summary
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step+1}")
        print(f"Network: {len(env.G.nodes())} nodes, {len(env.G.edges())} edges")
        print(f"Entropy: {info['entropy']:.2f}, Cost: {info['cost']:.2f}, Reliability: {info['reliability']:.2f}")
        print(f"Total Pipe Length: {env.calculate_total_pipe_length():.2f}")
        print("-" * 50)
        
        # Visualize at intervals and after final episode
        if episode % visualize_interval == 0 or episode == num_episodes - 1:
            # Save network visualization
            network_path = os.path.join(output_dir, f"network_episode_{episode+1}.png")
            env.visualize_network(episode=episode+1, save_path=network_path)
            print(f"Network visualization saved to {network_path}")
            
            # Save learning curves
            curves_path = os.path.join(output_dir, f"learning_curves_episode_{episode+1}.png")
            env.plot_learning_curves(save_path=curves_path)
            print(f"Learning curves saved to {curves_path}")
    
    # Plot overall training progress
    plt.figure(figsize=(10, 6))
    plt.plot(all_episode_rewards, 'b-')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress by Episode')
    plt.grid(True)
    progress_path = os.path.join(output_dir, "overall_training_progress.png")
    plt.savefig(progress_path)
    plt.close()
    
    # Print final network statistics
    print("\n" + "="*50)
    print("FINAL NETWORK STATISTICS:")
    print("="*50)
    print(f"Total Nodes: {len(env.G.nodes())}")
    print(f"- Demand Nodes: {len(env.demand_nodes)}")
    print(f"- Connector Nodes: {len(env.connector_nodes)}")
    print(f"Total Edges (Pipes): {len(env.G.edges())}")
    print(f"Total Pipe Length: {env.calculate_total_pipe_length():.2f}")
    print(f"Network Cost: {env.calculate_cost():.2f}")
    print(f"Network Entropy: {env.calculate_entropy():.2f}")
    print(f"Network Reliability: {env.calculate_reliability():.2f}")
    print("="*50)
    
    # Display the final network
    print("\nDisplaying final network topology...")
    env.visualize_network()
    
    return agent, all_final_metrics

def main():
    # Create an initial graph with demand nodes
    G = nx.Graph()
    
    # Define demand nodes and their values
    node_demands = {
        'A': 100,
        'B': 150,
        'C': 200,
        'D': 120,
        'E': 180
    }
    
    # Add nodes with arbitrary positions
    G.add_node('A', pos=(1, 1))
    G.add_node('B', pos=(8, 2))
    G.add_node('C', pos=(3, 8))
    G.add_node('D', pos=(7, 7))
    G.add_node('E', pos=(5, 5))
    
    # Connect them minimally to ensure initial connectivity
    G.add_edge('A', 'E', length=5.0, diameter=1.0)
    G.add_edge('B', 'E', length=3.6, diameter=1.0)
    G.add_edge('C', 'E', length=4.2, diameter=1.0)
    G.add_edge('D', 'E', length=2.8, diameter=1.0)
    
    # Define cost weights
    cost_weights = {
        'entropy': 10.0, # Weight for entropy term
        'cost': 1.0, # Weight for cost term
        'reliability': 5.0, # Weight for reliability term
        'connectivity': 1000, # Penalty for disconnected network
        'max_cost': 1000, # Maximum allowable cost
        'connector_cost': 5.0  # Fixed cost for each connector node
    }
    
    # Create environment
    env = WaterNetworkEnv(G, node_demands, cost_weights)
    
    # Train agent
    start_time = time.time()
    agent, metrics = train_water_network_agent(
        env, 
        num_episodes=200,
        batch_size=64,
        max_steps=100,
        visualize_interval=20,
        output_dir='water_network_results'
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds.")
    
    # Save training metrics to CSV
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv('water_network_results/training_metrics.csv', index=False)
    print("Training metrics saved to 'water_network_results/training_metrics.csv'")

if __name__ == "__main__":
    main()