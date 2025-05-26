
"""

In this file, we take the initial networks and convert units to match. we then assign pipe diameters closest to those of the originial network from the discrete set

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import random
import wntr
from wntr.network.io import write_inpfile
import time
from wntr.graphics.network import plot_network

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# In Modify_nets.py

# Import necessary libraries
import os
import wntr
from wntr.network.io import write_inpfile
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# The convert_units and map_pipe_diameters functions are correct and do not need changes.
def convert_units(wn):
    units = wn.options.hydraulic.inpfile_units
    if units != 'CMH':
        wn.options.hydraulic.inpfile_units = 'CMH'
        units = 'CMH'
    return wn, units

def map_pipe_diameters(wn, discrete_diameters):
    for pipe_id, pipe in wn.pipes():
        original_diameter = pipe.diameter
        closest_diameter = min(discrete_diameters, key=lambda x: abs(x - original_diameter))
        pipe.diameter = closest_diameter
    return wn

def enable_pumps(wn):
    """
    This function adds time-based CONTROLS for each pump to modulate their
    speed according to a predefined pattern. This is the robust method
    that ensures the logic is written to the final .inp file.
    """
    # print("Enabling pumps with time-based speed controls...")
    pump_pattern = [0.8, 0.7, 0.7, 0.6, 0.6, 0.7,  # Hours 0-5
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # Hours 6-11
                    1.2, 1.2, 1.2, 1.3, 1.3, 1.2,   # Hours 12-17
                    1.1, 1.0, 1.0, 0.9, 0.8, 0.8]   # Hours 18-23
    
    wn.add_pattern('pump_pattern', pump_pattern)

    # Loop through each pump in the network
    pump_name = wn.pump_name_list[0]
    pump_1 = wn.get_link(pump_name)
    pump_1.speed_pattern_name = 'pump_pattern'

    # Add a base-speed to pump 1
    pump_1.base_speed = 1.0  # Set the base speed to 1.0 (100%)
    
    return wn

if __name__ == "__main__":
    
    files = ['anytown-3.inp', 'hanoi-3.inp']
    script = os.path.dirname(__file__)

    print("----------------------------------")

    for file in files:
        file_path = os.path.join(script, 'Initial_networks', 'exeter', file)
        save_path = os.path.join(script, 'Modified_nets', file)
        
        print(f"\nProcessing network: {file}")

        # 1. Load the model from the original file
        wn = wntr.network.WaterNetworkModel(file_path)
    
        # 2. Convert units in memory
        wn, units = convert_units(wn)

        # 3. Apply modifications if it's the Anytown network
        # if 'anytown' in file.lower():
        #     # Add time-based controls for pump speed
        #     wn = enable_pumps(wn)
            
            # Manually assign the efficiency curve to each pump
            # print("Assigning efficiency curves to pumps...")
            # efficiency_curve = wn.get_curve('E1')
            # for pump_name in wn.pump_name_list:
            #     pump = wn.get_link(pump_name)
            #     pump.efficiency = efficiency_curve
            #     # print(f"  > Set efficiency for pump '{pump_name}' to curve '{pump.efficiency.name}'")

        # 4. Map pipe diameters in memory
        print("Mapping pipe diameters...")
        discrete_diameters = [0.3048, 0.4064, 0.508, 0.609, 0.762, 1.016]
        wn = map_pipe_diameters(wn, discrete_diameters)

        # Assign global values here for the network
        wn.options.hydraulic.demand_model = 'DDA'
        wn.options.hydraulic.headloss = 'H-W'
        wn.options.hydraulic.accuracy = 0.001
        wn.options.energy.global_efficiency = 75.0
        wn.options.energy.global_price = 0.26  # £/kWh

        # 5. Run simulation on the fully modified model object
        results = run_epanet_simulation(wn)
        
        # 6. Evaluate and print performance metrics
        performance_metrics = evaluate_network_performance(wn, results)
        energy_consumption = performance_metrics['total_energy_consumption']
        pump_cost = performance_metrics['total_pump_cost']
        print(f"Total energy consumption: {energy_consumption:.2f} kWh/day")
        print(f"Total pump cost: £{pump_cost:.2f}/day")
        # print(f"Modified energy consumption for {file}: {energy_consumption:.2f} kWh")
        # print(f"Units of the network: {units}")

        # 7. Write the final, fully modified model to the file
        # print(f"Saving modified network to: {save_path}")
        write_inpfile(wn, save_path)