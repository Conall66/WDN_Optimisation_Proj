
"""
Actor and Critic Networks

Actor determines set of actions to take
Double critics determine rewards of those actions

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):

        """
        Input feature breakdown:

        state_dim = number of potential states the system can be i
        action_dim = number of potential actions the agent can make
        max_action = maximum value the environment allows for an action (given continuous action space)
            i.e. maximum length a branch can grow, minimum growth angle etc.
            Consider like a set of constraints
        hidden dimension = number of nodes in each hidden layer of the NN
            Default value of 256 allows decent level of complexity without increasing computational expense too far

        """

        super(Actor, self).__init__() # Tell the system this is our actor
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std required to describe normal distribution)
        
        self.max_action = max_action
        
    def forward(self, state): # feedforward network
        a = F.relu(self.l1(state)) # relu avoids vanishing gradient problem
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
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) # Add very small value to avoid zero error
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
        self.l3 = nn.Linear(hidden_dim, 1) # Final output value is the Q value (expected utility of a state,action pair)
        
        # Q2 architecture (for stability)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # computes using gpu
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2