
"""

In this file, we convert a network from a networkx graph to a .inp file. We feed this into the EPANET solver to determine hydraulic performance values, and return the pressure deficit and flow rate for each node in the network. We also return the total energy consumption of the network, which is the sum of the energy consumption of each pump in the network. These features will help determine the reward of the agent.

"""

# Import necessary libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import time
import wntr.metrics.economic as economics
import tempfile
import shutil

# convert networkx graph to .inp file

def convert_graph_to_wntr(graph):

    """
    Convert a networkx graph to a .inp file for EPANET simulation.

    Parameters:
    graph (networkx.Graph): The input graph representing the water network.

    Returns:
    The generated water distribution network model as .inp file.
    """

    print("Converting graph to water network model...")

    # Initialise water network model
    wn = wntr.network.WaterNetworkModel()

    # Metrics settings

    # Identify node types from graph, and add them to the water network model accordingly

    """Positions don't need to be passed, since all infomation about positions is encapsulated by connections and lengths of edges."""

    # for node, data in graph.nodes(data=True):
    #     if data['type'] == 'Junction':
    #         wn.add_junction(node, base_demand=data['demand'], elevation=data['elevation'])
    #     elif data['type'] == 'Reservoir':
    #         wn.add_reservoir(node, base_head=data['head']) # Head is the elevation here
    #     elif data['type'] == 'Tank':
    #         wn.add_tank(node, elevation=data['elevation'], init_level=data['init_level'], min_level=data['min_level'], max_level=data['max_level'])
    #     elif data['type'] == 'Pump':
    #         wn.add_pump(node, node1=data['node1'], node2=data['node2'], pump_parameter=data['pump_parameter'], pump_type=data['pump_type'])

    for node, data in graph.nodes(data=True):
        if data['type'] == 'reservoir':
            wn.add_reservoir(name = node, base_head = data['base_head'], coordinates = data['coordinates'])
        elif data['type'] == 'tank':
            wn.add_tank(name = node, elevation = data['elevation'], init_level = data['init_level'], min_level = data['min_level'], max_level = data['max_level'], diameter = data['diameter'], coordinates = data['coordinates'])
        elif data['type'] == 'commercial':
            wn.add_junction(name = node, base_demand = data['demand'], elevation = data['elevation'], coordinates = data['coordinates'])
        elif data['type'] == 'residential':
            wn.add_junction(name = node, base_demand = data['demand'], elevation = data['elevation'], coordinates = data['coordinates'])
        elif data['type'] == 'junction':
            wn.add_junction(name = node, base_demand = data['base_demand'], elevation = data['elevation'], coordinates = data['coordinates'])
        elif data['type'] == 'pump':
            wn.add_pump(name = node, start_node_name = data['start_node'], end_node_name = data['end_node'], pump_parameter = data['pump_parameter'], pump_type = data['pump_type'])

    # Add pipes to the water network model with assigned diameters and roughness values
    pipe_count = 1
    for u, v, data in graph.edges(data=True):
        wn.add_pipe(name = f"{pipe_count}", start_node_name = f"{u}", end_node_name = f"{v}", length=data['length'], diameter=data['diameter'])
        pipe_count += 1

    return wn

def convert_wntr_to_nx(wn, results=None):
    """
    Convert a WNTR model to a NetworkX graph compatible with visualise_network
    """
    G = nx.Graph()
    
    # Add nodes
    for node_name, node in wn.nodes():
        # Get coordinates
        coords = node.coordinates
        
        # Determine node type and add appropriate attributes
        if isinstance(node, wntr.network.elements.Junction):
            node_type = 'residential'  # Using residential as default for junctions
            G.add_node(node_name, 
                      type=node_type, 
                      coordinates=coords,
                      elevation=node.elevation,
                      base_demand=node.base_demand)
        
        elif isinstance(node, wntr.network.elements.Reservoir):
            G.add_node(node_name, 
                      type='reservoir', 
                      coordinates=coords,
                      elevation=0,
                      base_head=node.base_head)
            
        elif isinstance(node, wntr.network.elements.Tank):
            G.add_node(node_name, 
                      type='tank', 
                      coordinates=coords,
                      elevation=node.elevation,
                      init_level=node.init_level)
    
    # Add edges (pipes and pumps)
    for link_name, link in wn.links():
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        if isinstance(link, wntr.network.elements.Pipe):
            G.add_edge(start_node, end_node,
                      node_id=link_name,
                      length=link.length,
                      diameter=link.diameter,
                      roughness=link.roughness)
                      
        elif isinstance(link, wntr.network.elements.Pump):
            if link.pump_type == 'POWER':
                G.add_edge(start_node, end_node,
                          node_id=link_name,
                          pump_type=link.pump_type,
                          pump_parameter=link.pump_parameter)
            else:
                G.add_edge(start_node, end_node,
                          node_id=link_name,
                          pump_type=link.pump_type,
                          pump_curve=link.pump_curve_name)
    
    return G

def run_epanet_simulation(wn, static=False):
    """
    Run EPANET simulation on the water network model in a process-safe temporary directory
    with a robust cleanup mechanism to avoid file lock errors in parallel execution.
    """
    # --- Simulation options setup (no changes here) ---
    if static:
        wn.options.time.duration = 0
    else:
        wn.options.time.duration = 24 * 3600

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

    # --- MODIFICATION FOR ROBUST PARALLEL EXECUTION ---

    # 1. Store the original working directory.
    original_cwd = os.getcwd()
    
    # 2. Manually create a temporary directory instead of using a 'with' block.
    # This gives us full control over the cleanup process.
    temp_dir_path = tempfile.mkdtemp(prefix="wntr_sim_")
    
    results = None
    
    try:
        # 3. Change into the isolated directory to run the simulation.
        os.chdir(temp_dir_path)
        
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

    except Exception as e:
        print(f"ERROR: Exception during EPANET simulation in temp dir {temp_dir_path}: {e}")
        # The finally block will still run to perform cleanup.
        raise  # Re-raise the exception so the calling environment knows about the failure.

    finally:
        # 4. CRITICAL: Always change the CWD back to the original path first.
        os.chdir(original_cwd)
        
        # 5. Implement a robust cleanup with a retry loop.
        for i in range(5):  # Try to clean up 5 times
            try:
                shutil.rmtree(temp_dir_path)
                # If deletion is successful, break the loop
                break
            except PermissionError:
                # This catches the exact [WinError 32] you are seeing.
                # print(f"Cleanup attempt {i+1} failed for {temp_dir_path}. Retrying in 0.1s...")
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: An unexpected error occurred during cleanup of {temp_dir_path}: {e}")
                break
        else:
            # This 'else' block runs only if the 'for' loop completes without a 'break'.
            # This means all cleanup attempts failed.
            print(f"Warning: Could not clean up temporary directory {temp_dir_path}. It may need to be deleted manually.")

    # --- END OF MODIFICATION ---
    
    # The rest of the function remains the same.
    return results

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
    # Example usage
    
    # Load example network from modified_networks folder
    # example_network_path = os.path.join('Modified_nets', 'hanoi_densifying_3', 'Step_1.inp')

    path = os.path.join(os.path.dirname(__file__), 'Modified_nets', 'hanoi_sprawling_3')

    for file in os.listdir(path):
        if file.endswith('.inp'):
            example_network_path = os.path.join(path, file)
            print(f"Using network file: {example_network_path}")

            wn = wntr.network.WaterNetworkModel(example_network_path)
            results = run_epanet_simulation(wn)

            # Print nodal demand values and demand supplied values in a table
            # demand_data = []
            # for node, node_data in wn.junctions():
            #     node_name = node_data.name
            #     base_demand = node_data.base_demand
            #     demand_supplied = np.mean(results.node['demand'][node_name])
            #     pressure = np.mean(results.node['pressure'][node_name])
            #     demand_data.append({
            #         'Node': node_name,
            #         'Base Demand': base_demand,
            #         'Demand Supplied': demand_supplied,
            #         'Pressure (m)': pressure
            #     })
            # demand_df = pd.DataFrame(demand_data)
            # print("Nodal Demand and Pressure Data:")
            # print(demand_df)

            performance_metrics = evaluate_network_performance(wn, results)

            print(performance_metrics)