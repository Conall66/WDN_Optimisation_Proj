
"""

In the sample WDN database there exist some networks with unusual characteristics. In this file, we manually address these to make the systems more realistic. 

"""

"""
Errors in the networks:
- 6848-68 nodes: the reservoir is placed at an elevation of 100m whilst every other junction has an elevation of 0, causing massive headlosses in teh first pipes. Converting this to an elevation of 0 with a 100m pump solves the issue.
- anytown-3 all good
- elhray-zeroflow: reservoir elevated above active plane
- gorev2: no headloss in the pipes?
- hanoi-3: reservoir elevated above active plane
- mod_anytown: headloss pressures all negative (the reservoir is at 0m elevation)
- noflow: headloss values seem to be static
- sampletown: tank and reservoir above the active plane
- st-net3-3: disconnected reservoir?
- st-net3: disconnected reservoir?
- test101-3: looks good
- test101: looks good
- todoinitest: looks good


Tests:
- Check graph is connected (if not, identify disconnected components and connect to their closest node)
- Convert the base elevation of reservoirs to 0m and add a pump with head equal to the difference in elevation
- Check for any negative pressures in the network and if so, increase the pump capacity
- Flag any unexpected values in the results of the simulation
"""

# Import necessary libraries
import os
import wntr
from wntr.network.io import write_inpfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import shutil

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Visualise_network import visualise_network

# Flag any units inconsistencies amongst the networks

def check_units(inp_file):
    """
    Function to check for units inconsistencies in the input file.
    
    Parameters:
    - inp_file: str, path to the input file
    
    Returns:
    - None
    """

    wn = wntr.network.WaterNetworkModel(inp_file)
    
    # Check for units inconsistencies
    # Example: Check if all elevations are in meters
    flow_units = wn.options.hydraulic.inpfile_units
    pressure_units = wn.options.hydraulic.inpfile_pressure_units # by default None
    headloss_formula = wn.options.hydraulic.headloss
    
    return {
        'flow_units': flow_units,
        'pressure_units': pressure_units,
        'headloss_formula': headloss_formula,
    }
    

# Manually change the elevation of the reservoir in 6848-68 to 0m and add a pump with a head of 100m

def change_elevation_and_add_pump(inp_file):
    """
    
    This file takes as input an inp file for which the reservoir is elevated above the active plane, converts the elevation to 0m and adds a pump with a head equivalent to the difference in elevation

    Parameters:
    - inp_file: str, path to the input file

    Returns:
    - None

    """

    wn = wntr.network.WaterNetworkModel(inp_file)
    
    # Get the reservoir node
    reservoir_node = wn.reservoir_name_list[0] # Assuming there is only one reservoir
    print(f"Reservoir nodes: {reservoir_node}")
    
    # Get the elevation of the reservoir
    elevation = wn.get_node(reservoir_node).base_head
    
    # Set the elevation to 0m
    wn.get_node(reservoir_node).base_head = 10.0 # Giving a small amount of elevation to ensure water can flow through pump

    # Find the node connected to the reservoir
    G = wn.to_graph()
    connected_nodes = list(G.neighbors(reservoir_node))
    # Extract the first connected node
    connected_node = connected_nodes[0]

    # Add a pump with a head equal to the difference in elevation
    pump_head = elevation  # Difference in elevation
    wn.add_curve('PumpCurve', 'HEAD', [(0, pump_head), (1, pump_head)])
    wn.add_pump('Pump', reservoir_node, connected_node, 'HEAD', 'PumpCurve')
    
    # Save the modified network to a new file
    modified_file_path = inp_file.replace('.inp', '_modified.inp')
    write_inpfile(wn, modified_file_path)
    
def reconnect_graph(inp_file):
    """
    Function to reconnect the graph if it is disconnected.
    
    Parameters:
    - inp_file: str, path to the input file
    
    Returns:
    - wn: WaterNetworkModel object
    """
    
    wn = wntr.network.WaterNetworkModel(inp_file)
    
    # Convert to networkx graph to check connectivity
    G = wn.get_graph()
    if not nx.is_connected(G):
        # Identify disconnected components
        components = list(nx.connected_components(G))
        
        # Connect the components to their closest node
        for component in components:
            if len(component) > 1:
                continue
            else:
                # Get the closest node to the component
                closest_node = min(component, key=lambda x: wn.get_node(x).elevation)
                # Connect the component to the closest node
                wn.add_pipe('Pipe', closest_node, component[0], length=10, diameter=0.1)
    
    return wn

def check_pressures(inp_file):

    """
    Function to check for negative pressures in the network.
    
    Parameters:
    - inp_file: str, path to the input file
    
    Returns:
    - wn: WaterNetworkModel object
    """
    
    wn = wntr.network.WaterNetworkModel(inp_file)
    
    # Get the hydraulic results
    results = run_epanet_simulation(wn)
    
    # Check for negative pressures
    pressures_below_crit = []
    for node in wn.junction_name_list:
        pressure = min(results.node['pressure'][node])
        if pressure < wn.options.hydraulic.required_pressure: # Global defined as 0.07 (water pressure)
            # Add the node to the list of noes with prssures below the critical value
            pressures_below_crit.append(node)      
    
    return pressures_below_crit

def check_headloss(inp_file):
    """
    Function to check for headloss in the network.
    
    Parameters:
    - inp_file: str, path to the input file
    
    Returns:
    - wn: WaterNetworkModel object
    """
    
    wn = wntr.network.WaterNetworkModel(inp_file)
    
    # Get the hydraulic results
    results = run_epanet_simulation(wn)
    
    # Check for headloss
    negative_headlosses = []
    for link in wn.pipe_name_list:
        # Check if the headloss value for a link is a list, otherwise return the value
        link_headloss = results.link['headloss'][link]

        if isinstance(link_headloss, pd.Series):
            headloss = min(link_headloss)
        else:
            headloss = link_headloss
            
        if headloss < 0:
            # Add the link to the list of links with headloss below the critical value
            negative_headlosses.append(link)
    
    return negative_headlosses

# Create report of network failures for each network and save to Initial_networks folder
def create_report(inp_file, save_directory, file_name):
    """
    Function to create a report of network failures.
    
    Parameters:
    - inp_file: str, path to the input file
    - save_directory: str, path to the directory where results will be saved
    - fil_name: str, name of the input file (without extension)
    
    Returns:
    - None
    """
    
    # Check for units inconsistencies
    units = check_units(inp_file)
    
    # Check for negative pressures
    pressures_below_crit = check_pressures(inp_file)
    
    # Check for headloss
    headloss = check_headloss(inp_file)
    
    # Create report
    report = {
        'file_name': file_name,
        'units': units,
        'pressures_below_crit': pressures_below_crit,
        'headloss': headloss
    }

    return report

# -------------------------------------------------------------------

if __name__ == "__main__":
    # Get the script directory
    script = os.path.dirname(__file__)
    report_file = os.path.join(script, 'Initial_networks', 'Initial_network_report.csv')
    input_file_path = os.path.join(script, 'Initial_networks')
    
    # Initialize a list to store all reports
    all_reports = []
    
    # Correctly iterate through files in the directory
    for root, dirs, files in os.walk(input_file_path):
        for file in files:
            if file.endswith('.inp'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0] # Get file name without extension
                
                # print(f"Processing {file_name}...")
                report = create_report(file_path, input_file_path, file_name)
                all_reports.append(report)
    
    # Convert all reports to DataFrame and save to CSV
    if all_reports:
        # Extract and process the data
        report_data = {
            'file_name': [],
            'flow_units': [],
            'pressure_units': [],
            'headloss_formula': [],
            'negative_pressure_count': [],
            'negative_headloss_count': []
        }
    
        for report in all_reports:
            if 'error' in report:
                continue
                
            report_data['file_name'].append(report['file_name'])
            
            # Extract units information
            units = report['units']
            report_data['flow_units'].append(units.get('flow_units', 'N/A'))
            report_data['pressure_units'].append(units.get('pressure_units', 'N/A'))
            report_data['headloss_formula'].append(units.get('headloss_formula', 'N/A'))
            
            # Count negative pressures and headlosses
            report_data['negative_pressure_count'].append(len(report['pressures_below_crit']))
            report_data['negative_headloss_count'].append(len(report['headloss']))
        
        # Create DataFrame and save to CSV
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_file, index=False)
        # print(f"Report saved to {report_file}")
    # else:
    #     print("No reports generated. Check if there are .inp files in the directory.")


    # ------------------------------------------------------------------
    # Modify the networks
    # ------------------------------------------------------------------

    modified_dir = os.path.join(script, 'Modified_initial_networks')
    if not os.path.exists(modified_dir):
        os.makedirs(modified_dir)
        # print(f"Created directory: {modified_dir}")

    modified_report_file = os.path.join(script, 'Initial_networks', 'Modified_initial_network_report.csv')

    # Copy all files from the input directory to the modified directory
    for root, dirs, files in os.walk(input_file_path):
        for file in files:
            if file.endswith('.inp'):
                # Get source and destination paths
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(modified_dir, file)
                
                try:
                    # Copy the file
                    shutil.copy2(src_file_path, dst_file_path)
                    # print(f"Copied {file} to {modified_dir}")
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")

    # elhray, hanoi and sample town change the elevation of the reservoir
    mod_reservoir_height = ['Elhay-ZeroFlow.inp', 'hanoi-3.inp', 'sampletown.inp', '6848-68nodes.inp']
    for file in mod_reservoir_height:
        file_path = os.path.join(modified_dir, file)
        # Check the file exists
        if os.path.exists(file_path):
            # Change the elevation of the reservoir to 0m and add a pump with a head of 100m
            change_elevation_and_add_pump(file_path)
        else:
            print(f"File {file} does not exist in the modified directory.")

    all_new_reports = []

    modified_net_vis = os.path.join(script, 'Modified_network_visualisation')

    for root, dirs, files in os.walk(modified_dir):
        for file in files:
            if file.endswith('.inp'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0] # Get file name without extension
                
                # print(f"Processing {file_name}...")
                # report = create_report(file_path, modified_dir, file_name)
                # all_new_reports.append(report)

                wn = wntr.network.WaterNetworkModel(file_path)

                # Run hydraulic analysis on the modified networks
                results = run_epanet_simulation(wn)
                # Evaluate network performance
                metrics = evaluate_network_performance(wn, results)
                # Save a network visualisation to the specified directory with file name
                visualise_network(wn, results, title=f"Water Distribution Network {file_name}", save_path=os.path.join(modified_net_vis, f"{file_name}.png"), mode='3d')
    
    # Convert all reports to DataFrame and save to CSV
    if all_new_reports:
        # Extract and process the data
        report_data = {
            'file_name': [],
            'flow_units': [],
            'pressure_units': [],
            'headloss_formula': [],
            'negative_pressure_count': [],
            'negative_headloss_count': []
        }
    
        for report in all_new_reports:
            if 'error' in report:
                continue
                
            report_data['file_name'].append(report['file_name'])
            
            # Extract units information
            units = report['units']
            report_data['flow_units'].append(units.get('flow_units', 'N/A'))
            report_data['pressure_units'].append(units.get('pressure_units', 'N/A'))
            report_data['headloss_formula'].append(units.get('headloss_formula', 'N/A'))
            
            # Count negative pressures and headlosses
            report_data['negative_pressure_count'].append(len(report['pressures_below_crit']))
            report_data['negative_headloss_count'].append(len(report['headloss']))
        
        # Create DataFrame and save to CSV
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(modified_report_file, index=False)
        # print(f"Report saved to {modified_report_file}")
    # else:
    #     print("No reports generated. Check if there are .inp files in the directory.")

    
