
import networkx as nx
import wntr
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def generate_water_network(num_nodes=20, connection_probability=0.1, seed=2):
    """
    Generate a random water distribution network using NetworkX.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes in the network
    connection_probability : float
        Probability of edge connection between nodes
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    nx.Graph : The generated water network
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate a random geometric graph
    # Nodes are placed randomly in a unit square and edges connect nodes within a certain distance
    pos = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
    G = nx.random_geometric_graph(num_nodes, connection_probability + 0.15, pos=pos)
    
    # Ensure the graph is connected (water networks must be connected)
    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) < num_nodes:
        # Add edges to connect all components
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            node1 = random.choice(list(largest_cc))
            node2 = random.choice(list(components[i]))
            G.add_edge(node1, node2)
    
    # Ensure the graph remains connected
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Adding minimal spanning tree.")
        # Create a complete graph
        complete_graph = nx.complete_graph(num_nodes)
        # Add edges to make it connected
        for u, v in nx.minimum_spanning_tree(complete_graph).edges():
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    
    return G, pos

def convert_to_wntr_model(G, pos, elevation_range=(0, 100), pipe_diameter_range=(200, 400), 
                         roughness_range=(90, 140)):
    """
    Convert NetworkX graph to WNTR water network model.
    
    Parameters:
    -----------
    G : NetworkX graph
        The water network topology
    pos : dict
        Node positions
    elevation_range : tuple
        Range of node elevations (min, max)
    pipe_diameter_range : tuple
        Range of pipe diameters in mm (min, max)
    roughness_range : tuple
        Range of pipe roughness coefficients (min, max)
        
    Returns:
    --------
    wntr.network.WaterNetworkModel : WNTR water network model
    """
    # Create a water network model
    wn = wntr.network.WaterNetworkModel()
    
    # Set up reservoir (source) at the first node
    reservoir_node = 0
    wn.add_reservoir('reservoir', base_head=120, coordinates=pos[reservoir_node])
    
    # Set up tank (storage) at one of the furthest nodes from reservoir
    # Find the node furthest from the reservoir
    path_lengths = nx.single_source_shortest_path_length(G, reservoir_node)
    furthest_node = max(path_lengths, key=path_lengths.get)
    
    tank_elevation = np.random.uniform(elevation_range[0], elevation_range[1])
    wn.add_tank('tank', elevation=tank_elevation, init_level=20, min_level=0,
                max_level=40, diameter=20, coordinates=pos[furthest_node])
    
    # Add junctions (nodes)
    for i in G.nodes():
        if i != reservoir_node and i != furthest_node: # Furthest node is a tank
            elevation = np.random.uniform(elevation_range[0], elevation_range[1])
            junction_name = f'J{i}'
            base_demand = np.random.uniform(0.001, 0.01)  # m³/s
            wn.add_junction(junction_name, base_demand=base_demand, elevation=elevation, 
                           coordinates=pos[i])
    
    # Add pipes (edges)
    for i, (u, v) in enumerate(G.edges()):
        # Convert node indices to node names
        node1 = 'reservoir' if u == reservoir_node else ('tank' if u == furthest_node else f'J{u}')
        node2 = 'reservoir' if v == reservoir_node else ('tank' if v == furthest_node else f'J{v}')
        
        # Calculate pipe length based on node positions (scaled to meters)
        length = np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2) * 1000  # km to m
        
        # Assign random pipe properties
        diameter = np.random.uniform(pipe_diameter_range[0], pipe_diameter_range[1]) / 1000  # mm to m
        roughness = np.random.uniform(roughness_range[0], roughness_range[1])
        
        pipe_name = f'P{i}'
        wn.add_pipe(pipe_name, node1, node2, length=length, diameter=diameter, roughness=roughness)
    
    # Add a pump to ensure adequate pressure from reservoir to network
    pump_from_node = 'reservoir'
    # Find a node connected to the reservoir
    # Find a valid node connected to the reservoir
    pump_to_node = None
    for neighbor in G.neighbors(reservoir_node):
        candidate_node = f'J{neighbor}'
        if candidate_node in wn.junction_name_list:  # Ensure the node exists in the WNTR model
            pump_to_node = candidate_node
            break

    print(f"Adding pump from {pump_from_node} to {pump_to_node}")
    print(f"Junctions: {wn.junction_name_list}")
    print(f"Reservoir Node Neighbours: {list(G.neighbors(reservoir_node))}")
    
    # Add a pump
    wn.add_pump('pump1', pump_from_node, pump_to_node, 
               pump_parameter=50.0, pump_type='POWER') # Arbitrary pump power added
    
    # Add pump curve
    wn.add_curve('curve1', 'HEAD', [(0, 60), (0.01, 55), (0.02, 40)])
    
    return wn

def run_hydraulic_simulation(wn, duration=24*3600, hydraulic_timestep=3600):
    """
    Run a hydraulic simulation on the water network.
    
    Parameters:
    -----------
    wn : wntr.network.WaterNetworkModel
        The water network model
    duration : int
        Simulation duration in seconds
    hydraulic_timestep : int
        Hydraulic timestep in seconds
        
    Returns:
    --------
    tuple : (success boolean, simulation results object)
    """
    # Set simulation options
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = hydraulic_timestep
    wn.options.time.report_timestep = hydraulic_timestep

    # Indentify network features
    print(f"Nodes in the network: {wn.junction_name_list}")
    print(f"Edges in the network: {wn.pipe_name_list}")
    print(f"Junctions in the network: {wn.junction_name_list}")
    print(f"Pipes in the network: {wn.pipe_name_list}")
    print(f"Pumps: {wn.pump_name_list}")

    # Initialize simulator
    sim = wntr.sim.EpanetSimulator(wn)
    
    # Run simulation
    results = sim.run_sim()
    
    return True, results

def evaluate_network_performance(wn, results):
    """
    Evaluate the performance of the water network based on simulation results.
    
    Parameters:
    -----------
    wn : wntr.network.WaterNetworkModel
        The water network model
    results : wntr.sim.SimulationResults
        Results from the hydraulic simulation
        
    Returns:
    --------
    dict : Dictionary of performance metrics
    """
    metrics = {}
    
    # Pressure metrics
    pressure = results.node['pressure']
    metrics['min_pressure'] = pressure.min().min()
    metrics['max_pressure'] = pressure.max().max()
    metrics['avg_pressure'] = pressure.mean().mean()
    
    # Calculate percentage of nodes with pressure < 20 psi (inadequate)
    min_pressure_threshold = 20 / 14.5038  # Convert 20 psi to meters of head (1 psi ≈ 0.7 m)
    inadequate_pressure = (pressure < min_pressure_threshold).sum(axis=1)
    total_junctions = len(wn.junction_name_list)
    metrics['percent_nodes_inadequate_pressure'] = (inadequate_pressure.mean() / total_junctions) * 100
    
    # Flow metrics
    flow = results.link['flowrate']
    metrics['max_flow'] = flow.abs().max().max()
    metrics['avg_flow'] = flow.abs().mean().mean()
    
    # Velocity metrics (flow / area)
    velocity = {}
    for link_name, link in wn.links():
        if link.link_type == 'Pipe':
            area = np.pi * (link.diameter/2)**2
            if link_name in flow.columns:
                velocity[link_name] = flow[link_name].abs() / area
    
    if velocity:
        velocity_df = pd.DataFrame(velocity)
        metrics['max_velocity'] = velocity_df.max().max()
        metrics['avg_velocity'] = velocity_df.mean().mean()
        
        # Check for high velocity pipes (> 2 m/s)
        high_velocity = (velocity_df > 2).sum(axis=1)
        total_pipes = len(wn.pipe_name_list)
        metrics['percent_pipes_high_velocity'] = (high_velocity.mean() / total_pipes) * 100
    
    # Network resilience metrics
    try:
        # Compute network resilience based on pressure
        pressure_above_min = pressure.copy()
        min_required = 20 / 14.5038  # 20 psi converted to meters
        pressure_above_min = pressure_above_min.clip(lower=0)
        pressure_above_min = pressure_above_min.subtract(min_required).clip(lower=0)
        
        # Simplified resilience metric: average excess pressure above minimum
        metrics['pressure_resilience'] = pressure_above_min.mean().mean()
    except:
        metrics['pressure_resilience'] = "Could not compute"
    
    return metrics

def visualise_network(G, pos, results=None, title="Water Distribution Network"):
    """
    Visualize the water network with pressure or flow information if available.
    
    Parameters:
    -----------
    G : NetworkX graph
        The water network topology
    pos : dict
        Node positions
    wn : wntr.network.WaterNetworkModel
        The water network model
    results : wntr.sim.SimulationResults, optional
        Results from hydraulic simulation
    title : str
        Plot title
    """
    fig,ax = plt.subplots(figsize=(12, 8))
    
    # Create node color map based on node type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node == 0:  # Reservoir
            node_colors.append('blue')
            node_sizes.append(300)
        elif node == max(nx.single_source_shortest_path_length(G, 0), key=nx.single_source_shortest_path_length(G, 0).get):  # Tank
            node_colors.append('green')
            node_sizes.append(200)
        else:  # Junction
            node_colors.append('red')
            node_sizes.append(100)
    
    # Draw the network
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, 
            with_labels=True, font_weight='bold', edge_color='gray')
    
    print(f"Results Node Pressures: {results.node['pressure']}")
    
    # If results available, add pressure or flow information
    if results is not None:
        pressure = results.node['pressure'].mean()
        
        # Get junction pressures
        junction_pressures = {}
        for node in G.nodes():
            if node == 0:  # Reservoir
                junction_pressures[node] = 0  # Reservoirs don't have pressure
            elif node == max(nx.single_source_shortest_path_length(G, 0), key=nx.single_source_shortest_path_length(G, 0).get):  # Tank
                junction_pressures[node] = 0  # We don't show pressure for tanks
            else:
                junction_name = f'J{node}'
                if junction_name in pressure:
                    junction_pressures[node] = pressure[junction_name]
                else:
                    junction_pressures[node] = 0
        
        # Create a list of pressures for coloring
        node_pressures = [junction_pressures[node] for node in G.nodes()]
        
        # Draw the network with pressure information
        nx.draw(G, pos, node_color=node_pressures, cmap=plt.cm.coolwarm, 
                node_size=node_sizes, with_labels=True, font_weight='bold')
        
        # Add colorbar to the current figure
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(node_pressures), 
                                                                         vmax=max(node_pressures)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)  # Explicitly associate the colorbar with the axes
        cbar.set_label('Average Pressure (m)')
        ax.set_title(f"{title} - Pressure Distribution")

    else:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()

def main():
    # Step 1: Generate a random water distribution network
    print("Generating water distribution network...")
    G, pos = generate_water_network(num_nodes=10, connection_probability=0.15)
    
    # Step 2: Convert to WNTR model
    print("Converting to WNTR model...")
    wn = convert_to_wntr_model(G, pos)
    
    # Step 3: Run hydraulic simulation
    print("Running hydraulic simulation...")
    success, results = run_hydraulic_simulation(wn)

    print("Simulation completed successfully!")
    print(f"Results: {results}")
    print(f"Results node pressures: {results.node['pressure']}")
    
    # Step 4: Evaluate network performance
    print("Evaluating network performance...")
    import pandas as pd  # Import here to avoid dependency issues
    metrics = evaluate_network_performance(wn, results)
    
    # Print metrics
    print("\nNetwork Performance Metrics:")
    print("=" * 30)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Step 5: Visualise the network
    print("\nVisualising network...")
    visualise_network(G, pos, results)

if __name__ == "__main__":
    main()

# main()