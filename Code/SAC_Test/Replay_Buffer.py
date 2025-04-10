
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

"""
Replay Buffer

The replay buffer stores (s, a, r, s', d) - past experiences to avoid repetition

"""

class ReplayBuffer:
    def __init__(self, capacity = 1e6):
        self.buffer = deque(maxlen=capacity) # sets maximum number of trials to store in replay buffer at 1000000

    def add(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done)) # Adds (s, a, r, s', d) to replay buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # Generates random values to sample a range of actions across sample space (more efficient training avoids the need to sample every single iteration)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) #ungroup tuples and group by type (see below)

        """

        (s1, a1, r1, s'1, d1)       (s1, s2, s3, ..., sn)
        (s2, a2, r2, s'2, d2)  ...  (a2, a2, a3, ..., an)
        (sn, an, rn, s'n, dn)       (r2, r2, r3, ..., rn)

        """

        return state, action, reward, next_state, done

