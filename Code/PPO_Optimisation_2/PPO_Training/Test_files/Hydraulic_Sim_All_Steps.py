
"""

Test the hydraulic simulation works on all steps of each scenario

"""

"""
Test the hydraulic simulation works on all steps of each scenario
"""

import os
import pandas as pd
from copy import deepcopy
import sys
import wntr

# Add parent directory to path so we can import from parent modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

Net_folders = os.path.join(parent_dir, 'Modified_nets')

# Loop through items in Modified_nets folder
for item in os.listdir(Net_folders):
    item_path = os.path.join(Net_folders, item)
    
    # Check if the item is a directory (subfolder)
    if os.path.isdir(item_path):
        print(f"Processing scenario folder: {item}")
        
        # Now process files within this subfolder
        for file in os.listdir(item_path):
            if file.endswith('.inp'):
                file_path = os.path.join(item_path, file)
                wn = wntr.network.WaterNetworkModel(file_path)
                print(f"  Running simulation for {file} in {item}")
                try:
                    results = run_epanet_simulation(wn, file_path)
                    performance_metrics = evaluate_network_performance(wn, results)
                    print(f"  Demand satisfaction ratio: {performance_metrics['demand_satisfaction_ratio']:.4f}")
                except Exception as e:
                    print(f"  Error processing {file} in {item}: {e}")
                    break  # Changed from break to continue to process other files

        print("All files ran successfully")