
"""

In this script, we configure the setting for a hydraulic simulation and extract key performance metrics from the results

"""

import wntr
from wntr.network import WaterNetworkModel
import numpy as np
import pandas as pd

def run_epanet_sim(wn):
    """
    Run the hydraulic simulation using the WNTR library.
    
    Parameters:
    wn (WaterNetworkModel): The water network model to simulate.
    
    Returns:
    tuple: A tuple containing the hydraulic results and quality results.
    """

    # Set up the hydraulic simulation options
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.inpfile_units = 'CMH'
    wn.options.hydraulic.accuracy = 0.01
    wn.options.hydraulic.trials = 100
    wn.options.hydraulic.headloss = 'H-W'
    wn.options.hydraulic.demand_model = 'DDA'
    wn.options.energy.global_efficiency = 100.0
    wn.options.energy.global_price = 0.26

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results