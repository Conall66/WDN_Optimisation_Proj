
"""

Test that a hydraulic simulation can be run on a water network model with 0 diameter pipes

"""

import os
import sys
import wntr
from wntr.network import WaterNetworkModel

script = os.path.dirname(__file__)
# Go back to the parent directory
script = os.path.dirname(script)
# Define the path to the input file
sys.path.append(script)

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Visualise_network import visualise_network

file_path = os.path.join(script, 'Modified_nets', 'anytown_sprawling_3', 'Step_50.inp')
# Convert to wn model
wn = WaterNetworkModel(file_path)
# Run the hydraulic simulation
results = run_epanet_simulation(wn)
print(f"Hydraulic simulation completed for network")
# Evaluate the performance of the network
performance_metrics = evaluate_network_performance(wn, results)
print(f"Performance metrics generated for network")

# Visulaise the network
visualise_network(wn, results, title = 'Test Hydraulics', save_path= None, mode = '2d', show =True)

"""

You can't assign a diameter of 0 to a pipe or a roughness of 0, so arbitrary small values were assigned in the hope that the agent will know to upgrade these despite the cost incurred. From visualisation of anytown 3 with sprawling layout, the pressure at one particular node was very large. 

"""
