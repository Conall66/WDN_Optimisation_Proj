
"""

In this file, we generate an initial set of looped networks and check their hydraulic feasibility. This takes as inputs the network size, the network type and the uncertainty level to generate initial sets of networks with initial supply and demand values. These environment will be used as the starting points for training specific and general RL agents.

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def generate_looped_network(
        area_size,
        num_nodes, 
        num_reservoirs, 
        num_tanks,
        initial_level,
        min_level,
        max_level,
        tank_diameter,
        num_pumps,
        pump_type,
        pump_power, 
        roughness_coefficients, 
        base_demand, 
        elevation_map,
        pipes,
        elevation_range = [0, 100],
        environment_type = 'urban'):
    
    return graph