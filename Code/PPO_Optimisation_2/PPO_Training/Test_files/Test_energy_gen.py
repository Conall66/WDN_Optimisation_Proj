
# Import a .inp file, and run a simple hydraulic simulation to record energy consumption

import os
import wntr
from wntr.network import WaterNetworkModel

def run_epanet_simulation(wn):
    """
    Run a hydraulic simulation on the water network model and return the results.
    
    Parameters:
    wn (WaterNetworkModel): The water network model to simulate.
    
    Returns:
    wntr.sim.SimulationResults: The results of the simulation.
    """
    # Create a simulation object
    sim = wntr.sim.EpanetSimulator(wn)
    
    # Run the simulation
    results = sim.run_sim()

    # Extract the feature headings from results
    feature_headings = results.keys()
    print(f"Simulation completed. Results contain the following features: {feature_headings}")
    
    return results

def evaluate_network_performance(wn, results):
    """
    Evaluate the performance of the water network based on the simulation results.
    
    Parameters:
    wn (WaterNetworkModel): The water network model.
    results (wntr.sim.SimulationResults): The results of the simulation.
    
    Returns:
    dict: A dictionary containing performance metrics, including total energy consumption.
    """
    # Calculate total energy consumption
    total_energy_consumption = 0.0
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        energy = results.link_energy[pump_name].sum()
        total_energy_consumption += energy
    
    return {'total_energy_consumption': total_energy_consumption}

if __name__ == "__main__":
    # Define the path to the input file
    script = os.path.dirname(__file__)
    # Go back to the parent directory
    script = os.path.dirname(script)
    file_path = os.path.join(script, 'Initial_networks', 'exeter', 'anytown-3.inp')
    
    # Load the water network model
    wn = WaterNetworkModel(file_path)
    
    # Run the simulation
    results = run_epanet_simulation(wn)
    
    # # Evaluate performance metrics
    # performance_metrics = evaluate_network_performance(wn, results)
    # energy_consumption = performance_metrics['total_energy_consumption']
    
    # print(f"Total energy consumption: {energy_consumption:.2f} kWh")

