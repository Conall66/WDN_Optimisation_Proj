
"""

In this file, we define the environment for the PPO agent. This takes as inputs the network size, the network type and the uncertainty level to generate initial sets of networks with initial supply and demand values. These environment will be used as the starting points for training specific and general RL agents. 

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import math
import random
from datetime import datetime, timedelta

# Import functions from other files
from Demand_profiles import *
from Elevation_map import generate_elevation_map

Pipes = {
    'Pipe 1': {'diameter': 0.152, 'unit_cost': 68, 'carbon_emissions': 0.48},
    'Pipe 2': {'diameter': 0.203, 'unit_cost': 91, 'carbon_emissions': 0.59},
    'Pipe 3': {'diameter': 0.254, 'unit_cost': 113, 'carbon_emissions': 0.71},
    'Pipe 4': {'diameter': 0.305, 'unit_cost': 138, 'carbon_emissions': 0.81},
    'Pipe 5': {'diameter': 0.356, 'unit_cost': 164, 'carbon_emissions': 0.87},
    'Pipe 6': {'diameter': 0.406, 'unit_cost': 192, 'carbon_emissions': 0.96},
    'Pipe 7': {'diameter': 0.457, 'unit_cost': 219, 'carbon_emissions': 1.05},
    'Pipe 8': {'diameter': 0.508, 'unit_cost': 248, 'carbon_emissions': 1.14},
    'Pipe 9': {'diameter': 0.610, 'unit_cost': 305, 'carbon_emissions': 1.32}
    }
# In the format of {pipe_name: {'diameter': mm, 'unit_cost': USD/m, 'carbon_emissions': TonCO2/m}}

      