
"""
SAC class

Input model parameter to define learning rate, discount rate, temperature etc.
Define set of potential actions
Define update rules

"""

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

from Networks import Actor, Critic
from Replay_Buffer import ReplayBuffer

    #     self.critic_target.load_state_dict(self.critic.state_dict())
    #     self.critic_target.eval() # Set target critic to evaluation mode

    #     self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4) # Actor optimizer
    #     self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4) # Critic optimizer

    #     self.discount = discount # Discount factor for future rewards
    #     self.tau = tau # Soft update factor
    #     self.alpha = alpha # Temperature parameter for entropy
    #     self.learn_alpha = learn_alpha # Whether to learn the temperature parameter or not

    #     if learn_alpha:
    #         self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, dtype=torch.float32)
    #         self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     action = self.actor(state).cpu().data.numpy().flatten()
    #     return action

class SAC:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action, 
        discount=0.99, # Influence of future states (higher means greater influence of future states)
        tau=0.005, # Soft update factor
        alpha=0.2, # Standard val - default if learn alpha = False
        learn_alpha=True, # Change how much estimations affect value funct
        hidden_dim=256 # Standard
    ):
        
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4) # Actor optimizer
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4) # Critic optimizer

        self.discount = discount # Discount factor for future rewards
        self.tau = tau # Soft update factor
        self.alpha = alpha # Temperature parameter for entropy
        self.learn_alpha = learn_alpha # Whether to learn the temperature parameter or not

        # Automatic entropy tuning
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