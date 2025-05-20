
"""

In this file, we test certain functions before implementing them in the main code.

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import wntr
import math
import random

from Hydraulic_Model import *
from Visualise_network import *

def generate_test_graph_structures():

    # Generate 5 test graph
    graph1 = nx.random_graphs.erdos_renyi_graph(10, 0.5) # num nodes, prob of edge creation
    graph2 = nx.random_graphs.barabasi_albert_graph(10, 2) # num nodes, num edges to attach from a new node to existing nodes
    graph3 = nx.random_graphs.watts_strogatz_graph(10, 2, 0.5) # num nodes, num neighbours, prob of rewiring
    graph4 = nx.random_graphs.random_lobster(5, 0.2, 0.2) # num nodes, proba of adding edge to backbone, prob of adding edge one level beyond
    graph5 = nx.random_graphs.powerlaw_cluster_graph(10, 2, 0.5) # num nodes, num edges to attach from a new node to existing nodes, prob of rewiring

    # Visualise the 3 graphs in subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs[0, 0].set_title('Erdos-Renyi Graph')
    nx.draw(graph1, ax=axs[0, 0], with_labels=True)
    axs[0, 1].set_title('Barabasi-Albert Graph')
    nx.draw(graph2, ax=axs[0, 1], with_labels=True)
    axs[1, 0].set_title('Watts-Strogatz Graph')
    nx.draw(graph3, ax=axs[1, 0], with_labels=True)
    axs[1, 1].set_title('Random Lobster Graph')
    nx.draw(graph4, ax=axs[1, 1], with_labels=True)
    axs[1, 2].set_title('Powerlaw Cluster Graph')
    nx.draw(graph5, ax=axs[1, 2], with_labels=True)
    plt.tight_layout()
    plt.show()

    """

    From the tests above, the random lobster graphs and powerlaw clustrer graphs seem like good candidates to describe branched and looped networks respectively. The random lobster graphs create a backbone (that could be considered the main pipe) and then add edges to the backbone and one level beyond. The powerlaw cluster graph creates a scale-free network with a high clustering coefficient, which is characteristic of many real-world networks, including water distribution systems.

    """

# Generate test branched graph structure to run hydraulic model on

def generate_test_graph_structure():
    # Create an empty graph instead of grid graph
    G = nx.Graph()
    
    # Add a reservoir node in the top left corner
    G.add_node('R1', type='reservoir', elevation=100, base_head=100, coordinates=(0, 100))
    
    # Add a tank in the top right corner
    G.add_node('T1', type='tank', elevation=30, init_level=50, min_level=0, 
              max_level=100, diameter=10, coordinates=(100, 100))
    
    # Add junctions in the grid
    junction_nodes = {
        'J1': {'elevation': 100, 'base_demand': 10, 'coordinates': (0, 90)},
        'J2': {'elevation': 90, 'base_demand': 10, 'coordinates': (0, 80)},
        'J3': {'elevation': 80, 'base_demand': 10, 'coordinates': (20, 60)},
        'J4': {'elevation': 70, 'base_demand': 10, 'coordinates': (40, 40)},
        'J5': {'elevation': 60, 'base_demand': 10, 'coordinates': (60, 20)},
        'J6': {'elevation': 50, 'base_demand': 10, 'coordinates': (80, 20)},
        'J7': {'elevation': 40, 'base_demand': 10, 'coordinates': (100, 20)},
        'J8': {'elevation': 30, 'base_demand': 10, 'coordinates': (100, 40)},
        'J9': {'elevation': 20, 'base_demand': 10, 'coordinates': (100, 60)},
        'J10': {'elevation': 20, 'base_demand': 10, 'coordinates': (100, 80)}
    }
    
    for node, attrs in junction_nodes.items():
        G.add_node(node, type='junction', **attrs)
    
    Pipes = {
        'Pipe 1': {'diameter': 0.152, 'unit_cost': 68, 'carbon_emissions': 0.48},
        'Pipe 2': {'diameter': 0.203, 'unit_cost': 91, 'carbon_emissions': 0.59},
        'Pipe 3': {'diameter': 0.254, 'unit_cost': 113, 'carbon_emissions': 0.71},
        'Pipe 4': {'diameter': 0.305, 'unit_cost': 138, 'carbon_emissions': 0.81},
        'Pipe 5': {'diameter': 0.356, 'unit_cost': 164, 'carbon_emissions': 0.87},
        'Pipe 6': {'diameter': 0.406, 'unit_cost': 192, 'carbon_emissions': 0.96},
        'Pipe 7': {'diameter': 0.457, 'unit_cost': 219, 'carbon_emissions': 1.05},
        'Pipe 8': {'diameter': 0.508, 'unit_cost': 248, 'carbon_emissions': 1.14},
        'Pipe 9': {'diameter': 0.610, 'unit_cost': 305, 'carbon_emissions': 1.32}
    }
    
    roughness_values = [0.0001, 0.0002, 0.0003]

    ordered_nodes = ['R1', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'T1']
    for i in range(len(ordered_nodes) - 1):
        node1 = ordered_nodes[i]
        node2 = ordered_nodes[i + 1]
        node_id = f'Pipe {node1}{node2}'
        
        # Randomly select a pipe from the Pipes dictionary
        pipe = random.choice(list(Pipes.keys()))
        
        # Calculate length based on node coordinates
        coord1 = G.nodes[node1]['coordinates']
        coord2 = G.nodes[node2]['coordinates']
        length = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        diameter = Pipes[pipe]['diameter']
        unit_cost = Pipes[pipe]['unit_cost']
        carbon_emissions = Pipes[pipe]['carbon_emissions']
        roughness = random.choice(roughness_values)
        
        G.add_edge(node1, node2, node_id=node_id, length=length, diameter=diameter, 
                   cost=unit_cost * length, carbon_emissions=carbon_emissions, 
                   roughness=roughness)
    
    # Add a pump between the reservoir and the first junction
    # G.add_node('P1', type='pump', start_node='R1', end_node='J1',
    #           pump_parameter=50, pump_type='POWER')
    
    return G

# Generate test wntr model

def generate_test_wntr_model():

    wn = wntr.network.WaterNetworkModel()

    # Add a reservoir node
    wn.add_reservoir('R1', base_head=100, coordinates=(0, 100))
    # Add a tank node
    wn.add_tank('T1', elevation=30, init_level=50, min_level=0, max_level=100, diameter=10, coordinates=(100, 100))
    # Add junctions
    # Add junctions with MUCH LOWER demands (0.01 instead of 10)
    wn.add_junction('J1', base_demand=0.01, elevation=100, coordinates=(0, 90))
    wn.add_junction('J2', base_demand=0.01, elevation=90, coordinates=(0, 80))
    wn.add_junction('J3', base_demand=0.01, elevation=80, coordinates=(20, 60))
    wn.add_junction('J4', base_demand=0.01, elevation=70, coordinates=(40, 40))
    wn.add_junction('J5', base_demand=0.01, elevation=60, coordinates=(60, 20))
    wn.add_junction('J6', base_demand=0.01, elevation=50, coordinates=(80, 20))
    wn.add_junction('J7', base_demand=0.01, elevation=40, coordinates=(100, 20))
    wn.add_junction('J8', base_demand=0.01, elevation=30, coordinates=(100, 40))
    wn.add_junction('J9', base_demand=0.01, elevation=20, coordinates=(100, 60))
    wn.add_junction('J10', base_demand=0.01, elevation=20, coordinates=(100, 80))
    
    Pipes = {
        'Pipe 1': {'diameter': 0.152, 'unit_cost': 68, 'carbon_emissions': 0.48},
        'Pipe 2': {'diameter': 0.203, 'unit_cost': 91, 'carbon_emissions': 0.59},
        'Pipe 3': {'diameter': 0.254, 'unit_cost': 113, 'carbon_emissions': 0.71},
        'Pipe 4': {'diameter': 0.305, 'unit_cost': 138, 'carbon_emissions': 0.81},
        'Pipe 5': {'diameter': 0.356, 'unit_cost': 164, 'carbon_emissions': 0.87},
        'Pipe 6': {'diameter': 0.406, 'unit_cost': 192, 'carbon_emissions': 0.96},
        'Pipe 7': {'diameter': 0.457, 'unit_cost': 219, 'carbon_emissions': 1.05},
        'Pipe 8': {'diameter': 0.508, 'unit_cost': 248, 'carbon_emissions': 1.14},
        'Pipe 9': {'diameter': 0.610, 'unit_cost': 305, 'carbon_emissions': 1.32}
    }
    
    roughness_values = [0.0001, 0.0002, 0.0003]
    
    ordered_nodes = ['R1', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9','J10', 'T1']

    for i in range(len(ordered_nodes) - 1):
        node1 = ordered_nodes[i]
        node2 = ordered_nodes[i + 1]
        node_id = f'Pipe_{node1}_{node2}'
        
        # Randomly select a pipe from the Pipes dictionary
        pipe = random.choice(list(Pipes.keys()))
        
        # Calculate length based on node coordinates
        coord1 = wn.get_node(node1).coordinates
        coord2 = wn.get_node(node2).coordinates
        length = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        diameter = Pipes[pipe]['diameter'] * 1000  # Convert to mm
        unit_cost = Pipes[pipe]['unit_cost']
        carbon_emissions = Pipes[pipe]['carbon_emissions']
        roughness = random.choice(roughness_values)
        
        wn.add_pipe(name=node_id, start_node_name=node1, end_node_name=node2, 
                    length=length, diameter=diameter, roughness=roughness)
        
    # Add a pump with a proper pump curve
    wn.add_curve('pump_curve', 'HEAD', [[0, 100], [50, 80], [100, 60]])
    wn.add_pump('PUMP1', 'R1', 'J1', 'HEAD', 'pump_curve')
    
    # Run hydraulic simulation
    # Configure simulation options
    wn.options.time.duration = 0  # Steady state simulation
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600
    
    # Set hydraulic options for better convergence
    wn.options.hydraulic.accuracy = 0.01  # Set accuracy for hydraulic calculations
    wn.options.hydraulic.headloss = 'H-W'  # Hazen-Williams
    wn.options.hydraulic.demand_model = 'DDA'  # Demand-driven analysis is more stable
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Display the network
    # wntr.graphics.plot_network(wn)
    # plt.show()

    return wn, results

if __name__ == "__main__":
    
    # Generate test wntr model
    wn, results = generate_test_wntr_model()

    # Visualise the network
    G = wn.get_graph()
    # Get node coordinates
    pos = wn.query_node_attribute('coordinates')

    # Get pressure results for the first timestep (index 0)
    pressure = results.node['pressure'].iloc[0].to_dict()
    
    # Get headloss results for the first timestep
    headloss = results.link['headloss'].iloc[0].to_dict()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create node lists by type
    junction_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Junction']
    reservoir_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Reservoir']
    tank_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Tank']
    
    # Get edges (pipes and pumps)
    pipes = [link_name for link_name, link in wn.links() if link.link_type == 'Pipe']
    pumps = [link_name for link_name, link in wn.links() if link.link_type == 'Pump']
    
    # Draw the different types of edges
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(wn.get_link(p).start_node_name, wn.get_link(p).end_node_name) for p in pipes],
                          width=1.0, edge_color='gray', style='solid')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(wn.get_link(p).start_node_name, wn.get_link(p).end_node_name) for p in pumps],
                          width=2.0, edge_color='red', style='dashed')
    
    # Draw the different types of nodes with pressure-based coloring
    # Create normalized pressure values for color mapping (for junctions only)
    junction_pressure_values = [pressure[node] for node in junction_nodes]
    vmin = min(junction_pressure_values) if junction_pressure_values else 0
    vmax = max(junction_pressure_values) if junction_pressure_values else 100
    
    # Draw junctions with pressure-based coloring
    junction_nodes_collection = nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=junction_nodes, node_size=300, 
        node_color=junction_pressure_values, cmap=plt.cm.viridis,
        vmin=vmin, vmax=vmax, node_shape='o')
    
    # Add a colorbar for pressure - now with explicit axes reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, label='Pressure (m)')
    
    # Draw other nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=reservoir_nodes, node_size=500, node_color='blue', node_shape='s')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=tank_nodes, node_size=500, node_color='green', node_shape='^')
    
    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    
    # Add pressure values as text labels
    pressure_labels = {node: f"{pressure[node]:.1f}" for node in pressure if node in junction_nodes}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=pressure_labels, font_size=8, font_color='black',
                           font_weight='bold', verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Add headloss values as edge labels
    edge_labels = {}
    for pipe in pipes:
        start_node = wn.get_link(pipe).start_node_name
        end_node = wn.get_link(pipe).end_node_name
        if pipe in headloss:
            edge_labels[(start_node, end_node)] = f"{headloss[pipe]:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Junction'),
        mpatches.Patch(color='blue', label='Reservoir'),
        mpatches.Patch(color='green', label='Tank'),
        plt.Line2D([0], [0], color='gray', lw=1, label='Pipe'),
        plt.Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='Pump')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_title('Water Distribution Network with Pressures and Headlosses')
    ax.axis('off')
    plt.tight_layout()
    plt.show()