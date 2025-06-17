
"""

Test that if the diameters of all pipes were the maximum possible diameter provided by the pipes dictionary, that the pressures would all be above the pressure threshold

"""

import os
import sys
import wntr
import numpy as np
import unittest
import matplotlib.pyplot as plt

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

# Test how performance for either of the initial networks changes as the pipe diameters are iterated through
def test_diameter_performance(inp_file):
    """
    Test how performance for either of the initial networks changes as the pipe diameters are iterated through.
    """
    # Load the water network model
    wn = wntr.network.WaterNetworkModel(inp_file)

    pipe_diameters = [0.3048, 0.4064, 0.508, 0.609, 0.762, 1.016, 2.0]

    results_dict = {
        'diameters': pipe_diameters,
        'demand_satisfaction': [],
        'total_pressure': [],
        'pressure_deficit': [],
        'energy_consumption': [],
        'pump_cost': []
    }

    for diameter in pipe_diameters:
        for pipe, pipe_data in wn.pipes():
            # Change the diameter of each pipe to the current diameter
            pipe_data.diameter = diameter
            
        # Run simulation on new wn object
        results = run_epanet_simulation(wn)
        # Evaluate performance metrics
        metrics = evaluate_network_performance(wn, results)

        # Create a table to show how performance changes with diameter, with each row showing diameters and each column showing a feature of the performance metrics
        # print(f"Diameter: {diameter} m")

        results_dict['demand_satisfaction'].append(metrics['demand_satisfaction_ratio'])
        results_dict['total_pressure'].append(metrics['total_pressure'])
        results_dict['pressure_deficit'].append(metrics['total_pressure_deficit'])
        results_dict['energy_consumption'].append(metrics['total_energy_consumption'])
        results_dict['pump_cost'].append(metrics['total_pump_cost'])

        print(f"Diameter: {diameter} m, Demand Satisfaction Ratio: {metrics['demand_satisfaction_ratio']}, Total Pressure: {metrics['total_pressure']}, Pressure Deficit: {metrics['total_pressure_deficit']}, Energy Consumption: {metrics['total_energy_consumption']}, Pump Cost: {metrics['total_pump_cost']}")
        
        # print('---------------------------')

    return results_dict

def plot_performance_with_d(results_dict1, results_dict2, results_dict3, results_dict4, save_path, show = True):
    """
    For the initial and final networks, plot the performance metrics against the pipe diameters in one figure
    """

    # Create figure with 4 subplots, one for each performance metric
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Set global colour profile for the plots
    # Set colour profile for the plots

    fig.suptitle('Performance Metrics vs Pipe Diameter', fontsize=16)

    # Plot how the demand satisfaction ratio changes with diameter for each scenario
    ax1.plot(results_dict1['diameters'], results_dict1['demand_satisfaction'], label='Initial Anytown Network', marker='o')
    ax1.plot(results_dict2['diameters'], results_dict2['demand_satisfaction'], label='Final Anytown Network', marker='o')
    ax1.plot(results_dict3['diameters'], results_dict3['demand_satisfaction'], label='Initial Hanoi Network', marker='o')
    ax1.plot(results_dict4['diameters'], results_dict4['demand_satisfaction'], label='Final Hanoi Network', marker='o')
    ax1.set_title('Demand Satisfaction Ratio vs Pipe Diameter for Scenarios')
    ax1.set_xlabel('Pipe Diameter (m)')
    ax1.set_ylabel('Demand Satisfaction Ratio')
    ax1.legend()
    ax1.grid(True)

        # Plot how the pressure deficit changes with diameter for each scenario
    ax2.plot(results_dict1['diameters'], results_dict1['pressure_deficit'], label='Initial Anytown Network', marker='o')
    ax2.plot(results_dict2['diameters'], results_dict2['pressure_deficit'], label='Final Anytown Network', marker='o')
    ax2.plot(results_dict3['diameters'], results_dict3['pressure_deficit'], label='Initial Hanoi Network', marker='o')
    ax2.plot(results_dict4['diameters'], results_dict4['pressure_deficit'], label='Final Hanoi Network', marker='o')
    ax2.set_title('Pressure Deficit vs Pipe Diameter for Scenarios')
    ax2.set_xlabel('Pipe Diameter (m)')
    ax2.set_ylabel('Pressure Deficit (m)')
    ax2.legend()
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        plt.show()

if __name__ == "__main__":
    
    inp_file_1 = os.path.join(parent_dir, 'Modified_nets' ,'anytown-3.inp')
    inp_file_2 = os.path.join(parent_dir, 'Modified_nets', 'anytown_sprawling_3', 'Step_50.inp')
    inp_file_3 = os.path.join(parent_dir, 'Modified_nets', 'hanoi-3.inp')
    inp_file_4 = os.path.join(parent_dir, 'Modified_nets', 'hanoi_sprawling_3', 'Step_50.inp')

    # Test on the original hanoi network
    # inp_file_3 = os.path.join(parent_dir, 'Initial_networks', 'exeter', 'hanoi-3.inp')

    # Test the normal and worst case scenarios to ensure that the largest pipe sizes would not results in any pressures below threshold
    # print("Testing inp_file_1")
    # test_max_diameter_performance(inp_file_1)
    # print("------------------------------")
    # print("Testing inp_file_2")
    # test_max_diameter_performance(inp_file_2)
    # print("------------------------------")
    # print("Testing inp_file_3")
    # test_max_diameter_performance(inp_file_3)
    # print("------------------------------")
    # print("Testing inp_file_4")
    # test_max_diameter_performance(inp_file_4)

    """
    
    In any scenario, if the diameters are large enough the pressure deficit is small. Interestingly, for large pipes in the Hanoi network as an example, once the diameters are large enough then the demand satisfaction raito remains effectively unchanged - the demand might increase over time but the system is unincumbered by the pipe diameter but rather than pressure supplied - this is why, in the larger network, the pressure is much lower.
    
    """

    # Test the performance of each diameter
    # print("Testing diameter performance for inp_file_1")
    # test_diameter_performance(inp_file_1)
    # print("------------------------------")
    # print("Testing diameter performance for inp_file_2")
    # test_diameter_performance(inp_file_2)
    # print("------------------------------")

    # Plot the performance metrics against the pipe diameters for each scenario
    dict1 = test_diameter_performance(inp_file_1)
    dict2 = test_diameter_performance(inp_file_2)
    dict3 = test_diameter_performance(inp_file_3)
    dict4 = test_diameter_performance(inp_file_4)

    # Plot the performance metrics against the pipe diameters for each scenario
    save_path = os.path.join(parent_dir, 'Plots', 'Tests', 'Test_Diameter_Performance.png')
    plot_performance_with_d(dict1, dict2, dict3, dict4, save_path=save_path, show=True)