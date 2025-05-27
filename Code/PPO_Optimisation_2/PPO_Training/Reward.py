
"""

In this file we take the hydraulic simulation results from a network and other global features to determine the reward at a particular time step

"""

# Import libraries

import numpy as np
import wntr
import os
import random
import math
import pandas as pd

def calculate_reward(actions, performance_metrics):

    """
    Calculate the reward based on the performance metrics and actions taken.
    """

    # Reward takes the pressures of the network and the pressure deficit ot normalise the pressure_deficit_factor

    Total_pressure = performance_metrics['total_pressure'] 
    Pressure_deficit_sum = performance_metrics['total_pressure_deficit']
    # Normalise pressure deficit
    

    # Punish spending - takes the total cost of implementing changes as the combination of the pipes installed * unit cost, pump running cost and service cost for each pipe changed - normalises cost bsaed on maximum possible spending (service cost of modifying all pipes, pump operating cost can't be avoided, maximum possible cost of upgrading pipes (to largest pipe diameter))

    # Punish unserviced demand - takes the serviced demand ratio from the performance metrics

    # Punish disconnecting commnuities - takes suggested actions in suggest order and evaluates whether each action would leave any nodes disconnected from the network. If so, harsh punishment is applied.

    # 

    return None
