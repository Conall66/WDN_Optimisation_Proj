
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
# from Visualise_2 import *
from Elevation_map import *
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
    wn.add_tank('T1', elevation=80, init_level=50, min_level=0, max_level=100, diameter=10, coordinates=(100, 100))
    # Add junctions
    # Add junctions with MUCH LOWER demands (0.01 instead of 10)
    wn.add_junction('J1', base_demand=0.01, elevation=100, coordinates=(0, 90))
    wn.add_junction('J2', base_demand=0.01, elevation=90, coordinates=(0, 80))
    wn.add_junction('J3', base_demand=0.01, elevation=80, coordinates=(20, 60))
    wn.add_junction('J4', base_demand=0.01, elevation=70, coordinates=(40, 40))
    wn.add_junction('J5', base_demand=0.01, elevation=60, coordinates=(60, 20))
    wn.add_junction('J6', base_demand=0.01, elevation=50, coordinates=(80, 20))
    wn.add_junction('J7', base_demand=0.01, elevation=40, coordinates=(100, 20))
    wn.add_junction('J8', base_demand=0.01, elevation=50, coordinates=(100, 40))
    wn.add_junction('J9', base_demand=0.01, elevation=60, coordinates=(100, 60))
    wn.add_junction('J10', base_demand=0.01, elevation=70, coordinates=(100, 80))
    
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
        roughness = int(random.choice(roughness_values))
        
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

# Generate random lobster graph and convert into wntr model
def generate_test_lobster_wntr():

    G = nx.random_graphs.random_lobster(10, 0.5, 0.2) # num nodes, proba of adding edge to backbone, prob of adding edge one level beyond
    # Assign positions to nodes based on position in G
    pos = nx.spring_layout(G)
    # Scale positions to fit in a 100x100 grid
    pos = {node: (x * 100, y * 100) for node, (x, y) in pos.items()}

    # Assing node types
    centrality = nx.betweenness_centrality(G)
    reservoir_candidates = sorted(centrality, key=centrality.get, reverse=True)[:2]
    tanks_candidates = [node for node, degree in G.degree() if degree == 1]
    edge_betweenness = nx.edge_betweenness_centrality(G)
    pump_candidates = sorted(edge_betweenness, key=edge_betweenness.get, reverse=True)[:3]

    # Create wntr model
    wn = wntr.network.WaterNetworkModel()

    # Create elevation map
    elevation_map, peaks = generate_elevation_map(area_size=(1000, 1000),
                                                  elevation_range=(0, 100), 
                                                  num_peaks=2, 
                                                  landscape_type='hilly')

    # Create reservoir from candidates
    reservoir_pos = random.choice(reservoir_candidates)
    # Get elevation value
    reservoir_elevation = elevation_map[int(pos[reservoir_pos][1]), int(pos[reservoir_pos][0])]
    wn.add_reservoir('R1', base_head=reservoir_elevation, coordinates=pos[reservoir_pos])

    # Create tank from candidates
    tank_pos = random.choice(tanks_candidates)
    # Get elevation value
    tank_elevation = elevation_map[int(pos[tank_pos][1]), int(pos[tank_pos][0])]
    wn.add_tank('T1', elevation=tank_elevation, init_level=50, min_level=0, max_level=100, diameter=10, coordinates=pos[tank_pos])

    # Create junctions
    for node in G.nodes():
        if node not in [reservoir_pos, tank_pos]:
            # Get elevation value
            elevation = elevation_map[int(pos[node][1]), int(pos[node][0])]
            wn.add_junction(f'J{node}', base_demand=0.01, elevation=elevation, coordinates=pos[node])
    
    # Create pipes
    pipe_counter = 1
    for edge in G.edges():
        node1, node2 = edge
        start_node_name = f'J{node1}' if node1 not in [reservoir_pos, tank_pos] else 'R1'
        end_node_name = f'J{node2}' if node2 not in [reservoir_pos, tank_pos] else 'T1'
        length = random.uniform(10, 100)  # Random length between 10 and 100 meters
        diameter = random.choice([0.152, 0.203, 0.254])  # Random diameter from the list
        roughness = random.choice([0.0001, 0.0002, 0.0003])
        wn.add_pipe(f"P{pipe_counter}", start_node_name=start_node_name, end_node_name=end_node_name,
                    length=length, diameter=diameter, roughness=roughness)
        pipe_counter += 1
        
    # Create pumps
    for pump in pump_candidates:
        node1, node2 = pump
        start_node_name = f'J{node1}' if node1 not in [reservoir_pos, tank_pos] else 'R1'
        end_node_name = f'J{node2}' if node2 not in [reservoir_pos, tank_pos] else 'T1'
        wn.add_pump(f'Pump_{node1}_{node2}', start_node_name=start_node_name, end_node_name=end_node_name,
                    pump_parameter=50, pump_type='POWER')
        
    # Run hydraulic simulation
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

    # Visualise the network
    # visualise_wntr(wn = wn, elevation_map = elevation_map, results = results, title="Random Lobster WNTR Model")
    return wn, results

def test_hydraulic_model(inp_graph):

    wn = wntr.network.WaterNetworkModel(inp_graph)

    # Run hydraulic simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    # Visualise the network
    visualise_network(wn = wn, results = results, title="Test WNTR Model", save_path="test.png", mode='2d')

if __name__ == "__main__":


    """From this test script, even the original version of the anytown network returns negative pressues and headloss values"""
    # script = os.path.dirname(__file__)
    # file_path = os.path.join(script, 'Imported_networks', 'epanet-example-networks', 'epanet-tests', 'exeter', 'anytown-3.inp')
    # # check file path exists
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"File {file_path} does not exist.")
    # test_hydraulic_model(file_path)

    """Some networks are performing as expected, but their configuration is odd"""
    # script = os.path.dirname(__file__)
    # file_path = os.path.join(script, 'Imported_networks', 'epanet-example-networks', 'epanet-tests', 'exeter', 'hanoi-3.inp')
    # # check file path exists
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"File {file_path} does not exist.")
    # test_hydraulic_model(file_path)

    script = os.path.dirname(__file__)
    file_path = os.path.join(script, 'Modified_initial_networks', 'hanoi-3_modified.inp')
    # check file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    test_hydraulic_model(file_path)




    