
import wntr
import networkx as nx
import random
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx

script = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script)
sys.path.append(parent_dir)

from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
from Elevation_map import generate_elevation_map
from Visualise_network import visualise_network


def generate_large_looped_wdn(grid_size=10, spacing=100, demand=0.01):
    wn = wntr.network.WaterNetworkModel()
    G = nx.grid_2d_graph(grid_size, grid_size)

    # Add diagonal connections for loops
    # for x in range(grid_size - 1):
    #     for y in range(grid_size - 1):
    #         G.add_edge((x, y), (x + 1, y + 1))
    #         G.add_edge((x + 1, y), (x, y + 1))

    # Add boundary loops (wrap-around)
    for i in range(grid_size):
        G.add_edge((i, 0), (i, grid_size - 1))
        G.add_edge((0, i), (grid_size - 1, i))

    # Add junctions
    for node in G.nodes:
        node_id = f"J_{node[0]}_{node[1]}"
        coord = (node[0] * spacing, node[1] * spacing)
        wn.add_junction(node_id, base_demand = random.randint(1, 10), coordinates=coord)

    # Add pipes
    for i, (u, v) in enumerate(G.edges):
        start = f"J_{u[0]}_{u[1]}"
        end = f"J_{v[0]}_{v[1]}"
        wn.add_pipe(f"P_{i}", start, end, length=spacing, diameter=0.3, roughness=100)

    # Add reservoir at one corner
    wn.add_reservoir("R1", base_head = 100, coordinates=(-100, -100))
    wn.add_pipe("PR1", "R1", "J_0_0", length=100, diameter=0.3, roughness=100)

    # Randomly remove 50% of the pipes and any pipes connected
    pipes_to_remove = random.sample(wn.pipe_name_list, int(len(wn.pipe_name_list) * 0.4))
    for pipe in pipes_to_remove:
        wn.remove_link(pipe)

    # Remove any completely disconnected junctions by first converting back to nx graph
    # wn = remove_disconnected_nodes(wn)

    wn = prune_unconnected_to_reservoir(wn)

    return wn

def remove_disconnected_nodes(wn):
    all_nodes = set(wn.node_name_list)
    connected_nodes = set()

    # Find all nodes still connected to pipes
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        connected_nodes.add(pipe.start_node_name)
        connected_nodes.add(pipe.end_node_name)

    # Identify nodes that are fully disconnected
    disconnected_nodes = all_nodes - connected_nodes

    # Remove those nodes
    for node_name in disconnected_nodes:
        wn.remove_node(node_name)

    print(f"Removed {len(disconnected_nodes)} disconnected nodes.")
    return wn

def prune_unconnected_to_reservoir(wn):
    """
    This function identifies all nodes that have a path to the reservoir
    and removes any nodes and pipes that don't.
    """
    
    G = wn.to_graph().to_undirected()
    reservoir_node = wn.reservoir_name_list[0] if wn.reservoir_name_list else None

    for node in wn.junction_name_list:
        if node != reservoir_node and not nx.has_path(G, reservoir_node, node):

            for pipe in wn.get_links_for_node(node):
                wn.remove_link(pipe)
            wn.remove_node(node)

    return wn

def apply_pipe_diameters_and_roughness(wn, pipes):
     # For each pipe in the network, randomly assign a diameter value from the pipes dictionary

    diams = [pipe['diameter'] for pipe in pipes.values()]

    for pipe in wn.pipe_name_list:
        # Randomly select a pipe diameter and roughness from the provided dictionary
        pipe_diameter = random.choice(diams) * 1000
        wn.get_link(pipe).diameter = pipe_diameter
        wn.get_link(pipe).roughness = random.randint(60, 150)

    return wn

# In the main section:

if __name__ == "__main__":

    wn = generate_large_looped_wdn(grid_size=10, spacing=100, demand=0.01)
    print("Large looped water distribution network generated.")

    # Assign pipe diameters and roughness values
    pipes = {
        'Pipe_1': {'diameter': 0.3048, 'unit_cost': 36.58},
        'Pipe_2': {'diameter': 0.4064, 'unit_cost': 56.32},
        'Pipe_3': {'diameter': 0.5080, 'unit_cost': 78.71},
        'Pipe_4': {'diameter': 0.6096, 'unit_cost': 103.47},
        'Pipe_5': {'diameter': 0.7620, 'unit_cost': 144.60},
        'Pipe_6': {'diameter': 1.0160, 'unit_cost': 222.62}
    }

    wn = apply_pipe_diameters_and_roughness(wn, pipes)

    # wntr.graphics.plot_network(wn, title="Large Looped Water Distribution Network", 
    #                             node_size=50)
    # plt.show()

    # Generate elevation map with size matching our network dimensions
    # First, get the bounds of our network to size the elevation map appropriately
    node_coords = [wn.get_node(node).coordinates for node in wn.node_name_list]
    if node_coords:
        x_coords, y_coords = zip(*node_coords)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add a buffer
        buffer = 100
        width = int(max_x - min_x) + buffer*2
        height = int(max_y - min_y) + buffer*2
        
        # Generate elevation map
        elevation_data = generate_elevation_map(
            area_size=(width, height), 
            elevation_range=(0, 100),
            num_peaks=5, 
            landscape_type='flat', 
            seed=1
        )
        
        # Unpack both returned values
        elevation_map, peak_data = elevation_data
        
        # Assign elevation values from elevation map to each node
        for node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            # Convert node coordinates to elevation map indices
            x, y = node.coordinates
            
            # Normalize coordinates to elevation map dimensions
            x_norm = int((x - min_x + buffer) * (elevation_map.shape[1] - 1) / width)
            y_norm = int((y - min_y + buffer) * (elevation_map.shape[0] - 1) / height)
            
            # Ensure indices are within bounds
            x_norm = max(0, min(x_norm, elevation_map.shape[1] - 1))
            y_norm = max(0, min(y_norm, elevation_map.shape[0] - 1))
            
            # Get elevation at this point
            elevation = elevation_map[y_norm, x_norm]
            
            # Set node elevation (handle differently for reservoirs vs. junctions)
            if node_name in wn.reservoir_name_list:
                # For reservoirs, set the base head
                node.base_head = elevation
            else:
                # For junctions, set the elevation
                node.elevation = elevation # Hard set reservoir elevation to 100m for max

    # Add a power pump to the pipe connecting the reservoir to the rest of the network
    reservoir_node = wn.reservoir_name_list[0] if wn.reservoir_name_list else None
    # Find the junction connected to the reservoir
    if reservoir_node:
        for pipe_name in wn.get_links_for_node(reservoir_node):
            if wn.get_link(pipe_name).start_node_name == reservoir_node:
                start_node = wn.get_link(pipe_name).start_node_name
                end_node = wn.get_link(pipe_name).end_node_name
                wn.add_pump("Pump_1", start_node, end_node, pump_type='POWER', pump_parameter=50)
                break

    # Apply demand curve

    # Generate future networks

    # Initial visualisation
    wntr.graphics.plot_network(wn, title="Large Looped Water Distribution Network", 
                                node_size=50, node_attribute='elevation', add_colorbar=True)
    plt.show()
    
    # Run the hydraulic simulation
    # sim = wntr.sim.EpanetSimulator(wn)
    # results = sim.run_sim()

    results = run_epanet_simulation(wn)
    print("Hydraulic simulation completed.")

    # Visualise the network
    visualise_network(wn, results, 
                      title="Large Looped Water Distribution Network",
                      save_path=None, 
                      mode='2d', show=True)
