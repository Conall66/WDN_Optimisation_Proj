"""
This script provides core hydraulic modeling functions for the WNTR environment.
It is responsible for running EPANET simulations and evaluating network performance.

This version is REFACTORED for improved clarity, error handling, and robustness.
"""
from typing import Dict, Optional
import wntr
import pandas as pd
import wntr.sim.results
import numpy as np

def run_epanet_simulation(wn: wntr.network.WaterNetworkModel):
    """
    Runs an EPANET simulation for a given water network model.

    Args:
        wn: A wntr WaterNetworkModel object.

    Returns:
        A wntr Results object if the simulation is successful, otherwise None.
    """
    try:
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        # A basic check for valid results (e.g., at least one timestep of pressure data)
        if results is None or results.node['pressure'].empty:
            # print("Warning: Simulation ran but produced no valid results.")
            return None
        return results
    except Exception as e:
        # Catch potential errors during simulation (e.g., network is disconnected)
        # print(f"Error during EPANET simulation: {e}")
        return None

def evaluate_network_performance(wn, results, final_time=3600):
    """
    Evaluate the performance of the water network based on simulation results.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    results (wntr.sim.EpanetSimulator): The EPANET simulation results.
    final_time (int): The final time step to evaluate.

    Returns:
    dict: A dictionary containing the performance metrics and reward components.
    """

    # print("Evaluating network performance...")

    min_pressure = wn.options.hydraulic.required_pressure  # Minimum pressure required (in m)

    # Calculate pressure deficit and collect demand satisfaction data
    pressure_deficit = {}
    total_demand_met = 0
    total_demand_required = 0
    total_pressure = 0
    critical_pressure_violations = 0
    
    for node in wn.junction_name_list:
        pressure = np.mean(results.node['pressure'][node])
        total_pressure += pressure

        # Calculate pressure deficit
        if pressure < min_pressure:
            deficit = min_pressure - pressure
            pressure_deficit[node] = deficit
            critical_pressure_violations += 1
        
        # Calculate demand satisfaction
        node_obj = wn.get_node(node)
        required_demand = node_obj.base_demand
        if required_demand is not None and required_demand > 0:
            supplied_demand = np.mean(results.node['demand'][node])
            total_demand_required += required_demand
            
            # If pressure is adequate, consider demand met
            if pressure >= min_pressure:
                total_demand_met += min(supplied_demand, required_demand)

    # print(f"Required demand: {total_demand_required}, Met demand: {total_demand_met}")

    # Calculate demand satisfaction ratio
    demand_satisfaction_ratio = 1.0
    if total_demand_required > 0:
        demand_satisfaction_ratio = total_demand_met / total_demand_required
    
    # Calculate total pressure deficit
    total_pressure_deficit = sum(pressure_deficit.values())

    # Calculate total energy consumption using global efficiency
    total_energy_consumption = 0
    timestep_hours = wn.options.time.hydraulic_timestep / 3600.0
    
    # Use the global efficiency setting (not efficiency curves)
    efficiency = wn.options.energy.global_efficiency / 100.0  # Convert from percentage to decimal
    
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start_node = pump.start_node_name
        end_node = pump.end_node_name
        
        # For each timestep, calculate energy
        for t in results.node['head'].index:
            # Get flow at this timestep
            flow = results.link['flowrate'][pump_name][t]
            
            # Only calculate energy when pump is running (flow > 0)
            if flow > 0:
                # Calculate head difference across the pump
                head_start = results.node['head'][start_node][t]
                head_end = results.node['head'][end_node][t]
                head_diff = head_end - head_start
                
                # Calculate power in kW: P = ρgQH/η/1000
                # Where ρ = 1000 kg/m³, g = 9.81 m/s²
                power_kw = (1000 * 9.81 * flow * head_diff) / (efficiency * 1000)
                
                # Calculate energy in kWh: E = P × Δt
                energy_kwh = power_kw * timestep_hours
                total_energy_consumption += energy_kwh
    
    # Calculate cost
    price_per_kwh = wn.options.energy.global_price
    total_pump_cost = total_energy_consumption * price_per_kwh if price_per_kwh else 0
    
    return {
        'total_pressure': total_pressure,
        'total_pressure_deficit': total_pressure_deficit,
        'total_energy_consumption': total_energy_consumption,
        'total_pump_cost': total_pump_cost,
        'demand_satisfaction_ratio': demand_satisfaction_ratio
        # 'critical_pressure_violations': critical_pressure_violations,
    }

if __name__ == "__main__":

    # test_hydraulic.py
    # Path to one of your network files
    # Make sure the 'Modified_nets' folder is in the same directory
    network_path = 'Networks2/hanoi_densifying_1/Step_50.inp'

    print(f"Loading network: {network_path}")
    try:
        # 1. Load the network model
        wn = wntr.network.WaterNetworkModel(network_path)
        print("Network loaded successfully.")

        # 2. Run the EPANET simulation
        results = run_epanet_simulation(wn)
        if results:
            print("EPANET simulation ran successfully.")
            
            # 3. Evaluate the network performance
            metrics = evaluate_network_performance(wn, results)
            print("Network performance evaluated successfully.")
            print("\n--- Performance Metrics ---")
            for key, value in metrics.items():
                print(f"{key}: {value:.2f}")
            print("\n✅ Hydraulic module test passed!")
        else:
            print("\n❌ Hydraulic module test FAILED: Simulation did not produce results.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")