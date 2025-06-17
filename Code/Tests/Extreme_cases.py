
"""

In this script, we import the hanoi and anytown networks from the Networks2 folder and we test the pressure deficit values and demand satisfaction values as we increase the pipe diameters in the network

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import wntr
from wntr.network import WaterNetworkModel

hanoi_net = wntr.network.WaterNetworkModel(os.path.join('Networks2', 'hanoi-3.inp'))
anytown_net = wntr.network.WaterNetworkModel(os.path.join('Networks2', 'anytown-3.inp'))

def plot_pipe_inc(wn, pipe_diameters, title):
    """
    Plot the pressure deficit and demand satisfaction for a range of pipe diameters.
    
    Args:
        wn: WaterNetworkModel object.
        pipe_diameters: List of pipe diameters to test.
        title: Title for the plot.
    """
    pressure_deficits = []
    demand_satisfactions = []
    min_pressure = wn.options.hydraulic.required_pressure  # Minimum pressure required (in m)

    for diameter in pipe_diameters:
        # Update all pipe diameters
        for pipe in wn.pipe_name_list:
            wn.get_link(pipe).diameter = diameter
        
        # Run simulation
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        
        # Calculate total pressure deficit
        total_pressure_deficit = 0
        total_demand_met = 0
        total_demand_required = 0
        
        for node in wn.junction_name_list:
            pressure = np.mean(results.node['pressure'][node])
            
            # Calculate pressure deficit
            if pressure < min_pressure:
                deficit = min_pressure - pressure
                total_pressure_deficit += deficit
            
            # Calculate demand satisfaction
            node_obj = wn.get_node(node)
            required_demand = node_obj.base_demand
            if required_demand is not None and required_demand > 0:
                supplied_demand = np.mean(results.node['demand'][node])
                total_demand_required += required_demand
                
                # If pressure is adequate, consider demand met
                if pressure >= min_pressure:
                    total_demand_met += min(supplied_demand, required_demand)
        
        # Calculate demand satisfaction ratio as a percentage
        demand_satisfaction_ratio = 100.0  # Default to 100%
        if total_demand_required > 0:
            demand_satisfaction_ratio = (total_demand_met / total_demand_required) * 100
        
        pressure_deficits.append(total_pressure_deficit)
        demand_satisfactions.append(demand_satisfaction_ratio)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(pipe_diameters, pressure_deficits, marker='o')
    plt.title(f'Pressure Deficit vs Pipe Diameter - {title}')
    plt.xlabel('Pipe Diameter (m)')
    plt.ylabel('Pressure Deficit (m)')
    
    plt.subplot(1, 2, 2)
    plt.plot(pipe_diameters, demand_satisfactions, marker='o')
    plt.title(f'Demand Satisfaction vs Pipe Diameter - {title}')
    plt.xlabel('Pipe Diameter (m)')
    plt.ylabel('Demand Satisfaction (%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Define a range of pipe diameters to test
    PIPES_CONFIG = {
    'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58}, 
    'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
    'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71}, 
    'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
    'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60}, 
    'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
}
    pipe_diameters = [0.3048, 0.4064, 0.5080, 0.6096, 0.7620, 1.0160]  # in meters
    
    # Plot for the hanoi network
    plot_pipe_inc(hanoi_net, pipe_diameters, 'Hanoi Network')
    
    # Plot for the anytown network
    plot_pipe_inc(anytown_net, pipe_diameters, 'Anytown Network')
