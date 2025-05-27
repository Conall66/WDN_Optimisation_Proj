
"""

Test that if the diameters of all pipes were the maximum possible diameter provided by the pipes dictionary, that the pressures would all be above the pressure threshold

"""

import os
import sys
import wntr
import numpy as np
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) # Go back one step in directory
sys.path.append(parent_dir)

print(f"Script directory: {parent_dir}")

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Visualise_network import visualise_network

def test_max_diameter_performance(inp_file):
    """
    Test that if the diameters of all pipes were the maximum possible diameter provided by the pipes dictionary,
    that the pressures would all be above the pressure threshold.
    """
    # Load the water network model

    wn = wntr.network.WaterNetworkModel(inp_file)

    pipe_diameters = [0.3048, 0.4064, 0.508, 0.609, 0.762, 1.016]
    max_diameter = max(pipe_diameters)

    for pipe, pipe_data in wn.pipes():
        # Change the diameter of each pipe to the maximum diameter
        pipe_data.diameter = max_diameter
        
    # Change the diameters of all pipes to the maximum diameter in the pipes dictionary
    # Run simulation on new wn object
    results = run_epanet_simulation(wn)
    # Evaluate performance metrics
    performance_metrics = evaluate_network_performance(wn, results)

    # Visualise
    visualise_network(wn, results, title="Max Diameter Performance Test", save_path = None, mode = '3d', show = True)

    print(f"Performance Metrics: {performance_metrics}")

if __name__ == "__main__":
    
    inp_file_1 = os.path.join(parent_dir, 'Modified_nets' ,'anytown-3.inp')
    inp_file_2 = os.path.join(parent_dir, 'Modified_nets', 'anytown_sprawling_3', 'Step_50.inp')
    inp_file_3 = os.path.join(parent_dir, 'Modified_nets', 'hanoi-3.inp')
    inp_file_4 = os.path.join(parent_dir, 'Modified_nets', 'hanoi_sprawling_3', 'Step_50.inp')

    # Test the normal and worst case scenarios to ensure that the largest pipe sizes would not results in any pressures below threshold
    print("Testing inp_file_1")
    test_max_diameter_performance(inp_file_1)
    print("------------------------------")
    print("Testing inp_file_2")
    test_max_diameter_performance(inp_file_2)
    print("------------------------------")
    print("Testing inp_file_3")
    test_max_diameter_performance(inp_file_3)
    print("------------------------------")
    print("Testing inp_file_4")
    test_max_diameter_performance(inp_file_4)

    """
    
    In any scenario, if the diameters are large enough the pressure deficit is small
    
    """