
"""

In this file, we convert a network from a networkx graph to a .inp file. We feed this into the EPANET solver to determine hydraulic performance values, and return the pressure deficit and flow rate for each node in the network. We also return the total energy consumption of the network, which is the sum of the energy consumption of each pump in the network. These features will help determine the reward of the agent.

Units used throughout the simulation are as follows:

- Length: m
- Diameter: mm
- Roughness: mm
- Minor loss: m
- Pressure: m
- Flow rate: L/s
- Energy consumption: kWh
- Power: kW
- Elevation: m
- Time: s

"""

# Import necessary libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr

# convert networkx graph to .inp file

def convert_graph_to_wntr(graph):

    """
    Convert a networkx graph to a .inp file for EPANET simulation.

    Parameters:
    graph (networkx.Graph): The input graph representing the water network.

    Returns:
    The generated water distribution network model as .inp file.
    """

    # Initialise water network model
    wn = wntr.network.WaterNetworkModel()

    # Identify node types from graph, and add them to the water network model accordingly

    """Positions don't need ot be passed, since all infomation about positions is encapsulated by connections and lengths of edges."""

    for node, data in graph.nodes(data=True):
        if data['type'] == 'Junction':
            wn.add_junction(node, base_demand=data['demand'], elevation=data['elevation'])
        elif data['type'] == 'Reservoir':
            wn.add_reservoir(node, head=data['head'], elevation=data['elevation'])
        elif data['type'] == 'Tank':
            wn.add_tank(node, elevation=data['elevation'], init_level=data['init_level'], min_level=data['min_level'], max_level=data['max_level'])
        elif data['type'] == 'Pump':
            wn.add_pump(node, node1=data['node1'], node2=data['node2'], pump_curve=data['curve'])

    # Add pipes to the water network model with assigned diameters and roughness values
    for u, v, data in graph.edges(data=True):
        wn.add_pipe(name = f"{u}_{v}", start_node_name = f"{u}", end_node_name = f"{v}", length=data['length'], diameter=data['diameter'], roughness=data['roughness'])

    # Add pump to water network model
    for node, data in graph.nodes(data=True):
        if data['type'] == 'Pump':
            wn.add_pump(node, node1=data['node1'], node2=data['node2'], pump_curve=data['curve'])

    return wn

def run_epanet_simulation(wn, duration = (24*3600), time_step = 3600):

    """
    Run EPANET simulation on the water network model.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    duration (int): Duration of the simulation in seconds.
    time_step (int): Time step for the simulation in seconds.

    Simulations will be run to simulate a single day of operation, with a time step of 1 hour (3600 seconds). By minimising the runtime duration, we can accelarate model training.

    Returns:
    wntr.sim.EpanetSimulator: The EPANET simulator object.
    """

    # Initialise simulation parameters
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = time_step
    wn.options.time.quality_timestep = time_step

    # Create a simulator object
    sim = wntr.sim.EpanetSimulator(wn)

    # Run the simulation
    sim.run_simulation(duration=duration, time_step=time_step)

    results = sim.run_sim()

    return results

def evaluate_network_performance(wn, results):

    """
    Evaluate the performance of the water network based on simulation results.

    Parameters:
    wn (wntr.network.WaterNetworkModel): The water network model.
    results (wntr.sim.EpanetSimulator): The EPANET simulation results.

    Returns:
    dict: A dictionary containing the performance metrics.
    """

    # Calculate pressure deficit
    pressure_deficit = {}
    for node in wn.junctions():
        pressure = results.node['pressure'].loc[node]
        pressure_deficit[node] = max(0, 20 - pressure)  # Assuming a minimum pressure of 20 m

    # Calculate flow rate
    flow_rate = {}
    for pipe in wn.pipes():
        flow = results.link['flow'].loc[pipe]
        flow_rate[pipe] = flow

    # Calculate total energy consumption
    total_energy_consumption = sum(results.link['energy'].loc[pump] for pump in wn.pumps())

    return {
        'pressure_deficit': pressure_deficit,
        'flow_rate': flow_rate,
        'total_energy_consumption': total_energy_consumption
    }