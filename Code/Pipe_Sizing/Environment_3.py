
import networkx as nx
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from epyt import epanet

# Constants for realistic water network generation
PIPE_DIAMETERS = {
    'small': [50, 80, 100, 150],     # mm
    'medium': [150, 200, 250, 300],  # mm
    'large': [300, 400, 500, 600]    # mm
}

# Hazen-Williams roughness coefficient ranges
ROUGHNESS_RANGES = {
    'new_pipe': (130, 140),           # New pipes
    'good_condition': (110, 130),     # Good condition
    'normal': (90, 110),              # Normal condition
    'poor': (70, 90),                 # Poor condition
    'very_poor': (50, 70)             # Very poor condition
}

# Tank parameters
TANK_PARAMS = {
    'small': {'diameter': 3, 'min_level': 0, 'max_level': 3, 'initial_level': 1.5},
    'medium': {'diameter': 5, 'min_level': 0, 'max_level': 5, 'initial_level': 2.5},
    'large': {'diameter': 8, 'min_level': 0, 'max_level': 8, 'initial_level': 4.0}
}

# Junction demand patterns
DEMAND_PATTERNS = {
    'residential': [0.4, 0.3, 0.3, 0.4, 0.5, 0.8, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9,
                   0.8, 0.7, 0.6, 0.7, 0.8, 1.1, 1.4, 1.3, 1.2, 1.0, 0.8, 0.6],
    'industrial': [0.8, 0.7, 0.7, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.3, 1.3, 1.2,
                  1.2, 1.3, 1.3, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
}

def generate_grid_network(rows=3, cols=3, scale=100, jitter=0):
    """
    Generate a grid-based water network.
    
    Args:
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        scale (float): Scaling factor for node positions
        jitter (float): Random position variation to make the grid more realistic
        
    Returns:
        nx.Graph: A water distribution network
    """
    G = nx.Graph()
    
    # Create nodes with positions
    for i in range(rows):
        for j in range(cols):
            # Add some jitter to positions to make it more realistic
            jx = random.uniform(-jitter, jitter) * scale
            jy = random.uniform(-jitter, jitter) * scale
            
            node_id = i * cols + j
            # Store position for visualization
            G.add_node(node_id, pos=(j * scale + jx, i * scale + jy), 
                      elevation=random.uniform(0, 20))  # Random elevation
    
    # Create edges (pipes) - connect horizontally
    for i in range(rows):
        for j in range(cols - 1):
            node1 = i * cols + j
            node2 = i * cols + j + 1
            G.add_edge(node1, node2)
    
    # Connect vertically
    for i in range(rows - 1):
        for j in range(cols):
            node1 = i * cols + j
            node2 = (i + 1) * cols + j
            G.add_edge(node1, node2)

    # Randonly remove a couple of edges to create a more realistic network
    for _ in range(random.randint(1, 3)):
        u, v = random.choice(list(G.edges))
        G.remove_edge(u, v)

    # Remove one conector node and extend the pipe to the next node along the row
    if len(G.nodes) > 2:
        node_to_remove = random.choice([n for n in G.nodes if G.degree(n) > 1])
        neighbors = list(G.neighbors(node_to_remove))
        if len(neighbors) > 1:
            # Extend the pipe to the next node along the row
            for neighbor in neighbors:
                G.add_edge(node_to_remove, neighbor)
                G.remove_node(node_to_remove)
                break

    # Connect one node to another diagonally
    if len(G.nodes) > 2:
        node1 = random.choice(list(G.nodes))
        node2 = random.choice([n for n in G.nodes if n != node1 and G.has_edge(node1, n)])
        if node2:
            G.add_edge(node1, node2)

    # Assign random diameters and roughness coefficients to pipes
    for u, v in G.edges:
        diameter = random.choice(PIPE_DIAMETERS['medium'])  # mm
        roughness = random.uniform(*ROUGHNESS_RANGES['normal'])
        G.edges[u, v]['diameter'] = diameter
        G.edges[u, v]['roughness'] = roughness
        
        # Calculate pipe length based on node positions
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        length = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
        G.edges[u, v]['length'] = length  # meters
    
    return G

def generate_branched_network(num_nodes=15, branching_factor=0.3, scale=100):
    """
    Generate a branched water network that resembles a tree.
    
    Args:
        num_nodes (int): Total number of nodes
        branching_factor (float): Probability of branching at each node
        scale (float): Scaling factor for node positions
        
    Returns:
        nx.Graph: A water distribution network with a branched structure
    """
    G = nx.Graph()
    
    # Start with a root node
    G.add_node(0, pos=(0, 0), elevation=random.uniform(0, 20))
    
    next_node_id = 1
    active_nodes = [0]  # Nodes that can still branch
    
    while next_node_id < num_nodes and active_nodes:
        parent = random.choice(active_nodes)
        parent_pos = G.nodes[parent]['pos']
        
        # Number of children for this node
        num_children = random.randint(1, 3) if random.random() < branching_factor else 1
        num_children = min(num_children, num_nodes - next_node_id)
        
        for _ in range(num_children):
            if next_node_id >= num_nodes:
                break
                
            # Random direction from parent
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0.7, 1.3) * scale
            
            # New position
            x = parent_pos[0] + distance * np.cos(angle)
            y = parent_pos[1] + distance * np.sin(angle)
            
            # Add the node and edge
            G.add_node(next_node_id, pos=(x, y), elevation=random.uniform(0, 20))
            G.add_edge(parent, next_node_id)
            
            active_nodes.append(next_node_id)
            next_node_id += 1
        
        # Eventually remove parent from active nodes
        if random.random() < 0.3:
            active_nodes.remove(parent)
    
    return G

def generate_loop_network(num_nodes=15, connectivity=0.3, scale=100):
    """
    Generate a looped water network with cycles.
    
    Args:
        num_nodes (int): Total number of nodes
        connectivity (float): Higher values create more loops
        scale (float): Scaling factor for node positions
        
    Returns:
        nx.Graph: A water distribution network with loops
    """
    # First create a minimum spanning tree
    G = nx.random_geometric_graph(num_nodes, 0.3, dim=2)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Connect components
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            G.add_edge(node1, node2)
    
    # Scale positions
    for node in G.nodes:
        pos = G.nodes[node]['pos']
        G.nodes[node]['pos'] = (pos[0] * scale, pos[1] * scale)
        G.nodes[node]['elevation'] = random.uniform(0, 20)  # Random elevation
    
    return G

def assign_network_properties(G, network_type='medium', age_condition='normal', scale = 100):
    """
    Assign realistic properties to the network including pipe diameters,
    roughness coefficients, demands, etc.
    
    Args:
        G (nx.Graph): Water distribution network
        network_type (str): 'small', 'medium', or 'large'
        age_condition (str): 'new_pipe', 'good_condition', 'normal', 'poor', or 'very_poor'
        
    Returns:
        nx.Graph: Network with properties assigned
    """
    # Copy the graph to avoid modifying the original
    network = G.copy()
    num_nodes = len(network.nodes)
    
    # Select source node (reservoir/tank) - usually at the edge of the network
    source_candidates = [node for node in network.nodes if network.degree(node) <= 2]
    if not source_candidates:
        source_candidates = list(network.nodes)
    source_node = random.choice(source_candidates)
    
    # Assign node types and properties
    for node in network.nodes:
        if node == source_node:
            # This is a reservoir
            network.nodes[node]['type'] = 'reservoir'
            network.nodes[node]['head'] = random.uniform(30, 50)  # Total head (m)
        else:
            # This is a junction
            network.nodes[node]['type'] = 'junction'
            
            # Assign demand based on node position (more demand farther from source)
            source_pos = network.nodes[source_node]['pos']
            node_pos = network.nodes[node]['pos']
            distance = np.sqrt((node_pos[0] - source_pos[0])**2 + (node_pos[1] - source_pos[1])**2)
            
            # Scale demand with distance and add randomness
            base_demand = distance / (10 * scale) * random.uniform(0.5, 1.5)
            network.nodes[node]['demand'] = max(0.1, base_demand)  # L/s
            
            # Assign pattern type
            pattern_type = 'residential' if random.random() < 0.8 else 'industrial'
            network.nodes[node]['pattern'] = pattern_type
    
    # Randomly select a second node as a tank (elevated storage)
    non_source_nodes = [n for n in network.nodes if n != source_node]
    if len(non_source_nodes) > 5:  # Only add tank to larger networks
        tank_node = random.choice(non_source_nodes)
        network.nodes[tank_node]['type'] = 'tank'
        tank_size = network_type  # small, medium, or large
        for param, value in TANK_PARAMS[tank_size].items():
            network.nodes[tank_node][param] = value
        # Remove demand from tank
        if 'demand' in network.nodes[tank_node]:
            del network.nodes[tank_node]['demand']
    
    # Assign pipe properties
    for u, v in network.edges:
        # Select appropriate diameter based on network type
        diameter = random.choice(PIPE_DIAMETERS[network_type])
        network.edges[u, v]['diameter'] = diameter  # mm
        
        # Assign roughness coefficient based on condition
        min_rough, max_rough = ROUGHNESS_RANGES[age_condition]
        network.edges[u, v]['roughness'] = random.uniform(min_rough, max_rough)
        
        # Calculate pipe length based on node positions
        pos_u = network.nodes[u]['pos']
        pos_v = network.nodes[v]['pos']
        length = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
        network.edges[u, v]['length'] = length  # meters
    
    return network

def visualize_network(G, filename=None):
    """
    Visualize the water distribution network.
    
    Args:
        G (nx.Graph): Water distribution network
        filename (str, optional): If provided, save the figure to this file
    """
    plt.figure(figsize=(12, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes by type
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        node_type = G.nodes[node].get('type', 'junction')
        if node_type == 'reservoir':
            node_colors.append('blue')
            node_sizes.append(300)
        elif node_type == 'tank':
            node_colors.append('cyan')
            node_sizes.append(200)
        else:  # junction
            node_colors.append('red')
            node_sizes.append(100)
    
    # Draw edges (pipes) with width proportional to diameter
    edge_widths = []
    for u, v in G.edges:
        diameter = G.edges[u, v].get('diameter', 100)
        # Scale diameter for visualization
        edge_widths.append(diameter / 50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # Add node labels
    labels = {node: str(node) for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    
    plt.title("Water Distribution Network")
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
    plt.show()

def convert_to_epanet(G, filename):
    """
    Convert NetworkX graph to EPANET input file.
    
    Args:
        G (nx.Graph): Water distribution network with properties
        filename (str): Output EPANET .inp filename
        
    Returns:
        str: Path to the created EPANET file
    """

    d = epanet(filename)
    d.setTitle("Generated Water Distribution Network")
    d.setFlowUnitsLPS(1)  # L/s
    d.setOptionsHeadLossFormula(1)  # Hazen-Williams
    d.setOptionsSpecificGravity(1.0)  # Water
    d.setOptionsSpecificViscosity(1.0)  # Water
    # Set time step so simulation runs for 24 hours

    # Add nodes (junctions and tanks)
    for node in G.nodes:
        node_type = G.nodes[node].get('type', 'junction')
        if node_type == 'reservoir':
            d.addNodeReservoir(node, G.nodes[node]['head'], G.nodes[node]['elevation'])
        elif node_type == 'tank':
            d.addNodeTank(node, G.nodes[node]['diameter'], G.nodes[node]['min_level'],
                      G.nodes[node]['max_level'], G.nodes[node]['initial_level'],
                      G.nodes[node]['elevation'])
        else:  # junction
            demand = G.nodes[node].get('demand', 0)
            d.addNodeJunction(node, demand, G.nodes[node]['elevation'])

    # Add pipes
    for u, v in G.edges:
        diameter = G.edges[u, v].get('diameter', 100) / 1000  # Convert mm to m
        roughness = G.edges[u, v].get('roughness', 100)
        length = G.edges[u, v].get('length', 100)  # meters
        d.addLinkPipe(u, v, length, diameter, roughness)

    # Add patterns for junctions
    for node in G.nodes:
        if G.nodes[node].get('pattern'):
            pattern = G.nodes[node]['pattern']
            if pattern in DEMAND_PATTERNS:
                d.addPattern(node, DEMAND_PATTERNS[pattern])


    # Save the EPANET input file
    d.saveInputFile(filename)

    return filename

def run_epanet_simulation(inp_file):
    """
    Run an EPANET simulation and return hydraulic results.
    
    Args:
        inp_file (str): Path to EPANET input file
        
    Returns:
        dict: Dictionary with hydraulic results
    """
    # Load the input file
    d = epanet(inp_file)
    
    # Open the hydraulic solver
    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis()
    
    results = {
        'time': [],
        'pressures': {},
        'flows': {},
        'velocities': {}
    }
    
    # Initialize results structures
    for junction in d.getNodeJunctionNameID():
        results['pressures'][junction] = []
    
    for pipe in d.getLinkPipeNameID():
        results['flows'][pipe] = []
        results['velocities'][pipe] = []
    
    # Run step by step
    current_time = 0
    while True:
        # Get current time
        t = d.runHydraulicAnalysis()
        if t <= 0:
            break
            
        current_time = t
        results['time'].append(current_time)
        
        # Get junction pressures
        for junction in d.getNodeJunctionNameID():
            pressure = d.getNodePressure(junction)
            results['pressures'][junction].append(pressure)
        
        # Get pipe flows and velocities
        for pipe in d.getLinkPipeNameID():
            flow = d.getLinkFlowRate(pipe)
            velocity = d.getLinkVelocity(pipe)
            results['flows'][pipe].append(flow)
            results['velocities'][pipe].append(velocity)
    
    # Close the hydraulic solver
    d.closeHydraulicAnalysis()
    
    # Calculate some summary statistics
    summary = calculate_network_metrics(results)
    results['summary'] = summary
    
    return results

def calculate_network_metrics(results):
    """
    Calculate summary metrics from simulation results.
    
    Args:
        results (dict): Simulation results
        
    Returns:
        dict: Summary metrics
    """
    summary = {}
    
    # Average pressure
    all_pressures = []
    for junction, pressures in results['pressures'].items():
        all_pressures.extend(pressures)
    
    summary['avg_pressure'] = np.mean(all_pressures) if all_pressures else 0
    summary['min_pressure'] = np.min(all_pressures) if all_pressures else 0
    summary['max_pressure'] = np.max(all_pressures) if all_pressures else 0
    
    # Critical junctions (lowest pressure)
    junction_avg_pressures = {}
    for junction, pressures in results['pressures'].items():
        junction_avg_pressures[junction] = np.mean(pressures) if pressures else 0
    
    if junction_avg_pressures:
        min_pressure_junction = min(junction_avg_pressures, key=junction_avg_pressures.get)
        summary['critical_junction'] = min_pressure_junction
        summary['critical_junction_pressure'] = junction_avg_pressures[min_pressure_junction]
    
    # Flow metrics
    all_flows = []
    for pipe, flows in results['flows'].items():
        all_flows.extend(flows)
    
    summary['avg_flow'] = np.mean(all_flows) if all_flows else 0
    summary['total_flow'] = np.sum([np.mean(flows) for flows in results['flows'].values()]) if results['flows'] else 0
    
    # Velocity metrics
    all_velocities = []
    for pipe, velocities in results['velocities'].items():
        all_velocities.extend(velocities)
    
    summary['avg_velocity'] = np.mean(all_velocities) if all_velocities else 0
    summary['max_velocity'] = np.max(all_velocities) if all_velocities else 0
    
    # Count segments with velocity outside of desired range (0.5-2.0 m/s)
    count_low_velocity = 0
    count_high_velocity = 0
    for pipe, velocities in results['velocities'].items():
        avg_velocity = np.mean(velocities) if velocities else 0
        if avg_velocity < 0.5:
            count_low_velocity += 1
        elif avg_velocity > 2.0:
            count_high_velocity += 1
    
    summary['count_low_velocity_pipes'] = count_low_velocity
    summary['count_high_velocity_pipes'] = count_high_velocity
    
    return summary

def generate_water_network(network_topology='grid', network_size='medium', age_condition='normal', **kwargs):
    """
    Generate a complete water distribution network with all properties assigned.
    
    Args:
        network_topology (str): 'grid', 'branched', or 'loop'
        network_size (str): 'small', 'medium', or 'large'
        age_condition (str): 'new_pipe', 'good_condition', 'normal', 'poor', or 'very_poor'
        **kwargs: Additional parameters for specific topologies
        
    Returns:
        nx.Graph: Complete water distribution network
    """
    # Generate base network
    if network_topology == 'grid':
        rows = kwargs.get('rows', 4)
        cols = kwargs.get('cols', 4)
        G = generate_grid_network(rows=rows, cols=cols)
    elif network_topology == 'branched':
        num_nodes = kwargs.get('num_nodes', 15)
        G = generate_branched_network(num_nodes=num_nodes)
    elif network_topology == 'loop':
        num_nodes = kwargs.get('num_nodes', 15)
        G = generate_loop_network(num_nodes=num_nodes)
    else:
        raise ValueError(f"Unknown network topology: {network_topology}")
    
    # Assign properties
    G = assign_network_properties(G, network_type=network_size, age_condition=age_condition)
    
    return G

def analyse_water_network(G, output_dir="./output"):
    """
    analyse a water network by converting to EPANET and running simulation.
    
    Args:
        G (nx.Graph): Water distribution network
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Simulation results and metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate random network ID
    network_id = f"network_{random.randint(1000, 9999)}"
    
    # Visualize the network
    vis_file = os.path.join(output_dir, f"{network_id}_network.png")
    visualize_network(G, filename=vis_file)
    
    # Convert to EPANET
    inp_file = os.path.join(output_dir, f"{network_id}.inp") # Create a new empty EPANET file and pass to the function
    epanet_file = convert_to_epanet(G, inp_file)
    
    # Run simulation
    results = run_epanet_simulation(epanet_file)
    
    # Save results summary
    results_file = os.path.join(output_dir, f"{network_id}_results.txt")
    with open(results_file, 'w') as f:
        f.write("Water Network Analysis Results\n")
        f.write("=============================\n\n")
        
        f.write("Network Information:\n")
        f.write(f"- Nodes: {len(G.nodes)}\n")
        f.write(f"- Pipes: {len(G.edges)}\n\n")
        
        f.write("Simulation Results Summary:\n")
        for metric, value in results['summary'].items():
            f.write(f"- {metric}: {value}\n")
    
    # Return results
    return {
        'network': G,
        'epanet_file': epanet_file,
        'results': results,
        'visualization': vis_file,
        'summary_file': results_file
    }

def generate_multiple_networks(num_networks=3, output_dir="./output"):
    """
    Generate multiple water networks with varying properties.
    
    Args:
        num_networks (int): Number of networks to generate
        output_dir (str): Directory to save output files
        
    Returns:
        list: List of analysis results for each network
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i in range(num_networks):
        # Randomly choose network parameters
        topology = random.choice(['grid', 'branched', 'loop'])
        size = random.choice(['small', 'medium', 'large'])
        condition = random.choice(['new_pipe', 'good_condition', 'normal', 'poor'])
        
        # Additional parameters based on topology
        kwargs = {}
        if topology == 'grid':
            if size == 'small':
                kwargs = {'rows': random.randint(2, 3), 'cols': random.randint(2, 3)}
            elif size == 'medium':
                kwargs = {'rows': random.randint(3, 5), 'cols': random.randint(3, 5)}
            else:  # large
                kwargs = {'rows': random.randint(5, 7), 'cols': random.randint(5, 7)}
        else:
            if size == 'small':
                kwargs = {'num_nodes': random.randint(8, 12)}
            elif size == 'medium':
                kwargs = {'num_nodes': random.randint(12, 20)}
            else:  # large
                kwargs = {'num_nodes': random.randint(20, 30)}
        
        print(f"Generating network {i+1}/{num_networks}: {topology}, {size}, {condition}")
        
        # Generate network
        G = generate_water_network(network_topology=topology, network_size=size, 
                                  age_condition=condition, **kwargs)
        
        # analyse network
        network_results = analyse_water_network(G, output_dir)
        results.append(network_results)
        
        print(f"  - Nodes: {len(G.nodes)}, Pipes: {len(G.edges)}")
        print(f"  - Avg Pressure: {network_results['results']['summary']['avg_pressure']:.2f} m")
        print(f"  - Critical Junction: {network_results['results']['summary'].get('critical_junction', 'N/A')}")
        print()
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate and analyse a single network
    print("Generating a single water distribution network...")
    network = generate_water_network(network_topology='grid', network_size='medium')
    results = analyse_water_network(network)
    
    print("Network analysis complete.")
    print(f"EPANET file created: {results['epanet_file']}")
    print(f"Visualization saved: {results['visualization']}")
    print("\nSimulation summary:")
    for metric, value in results['results']['summary'].items():
        print(f"- {metric}: {value}")
    
    # Generate multiple networks
    print("\nGenerating multiple water distribution networks...")
    multiple_results = generate_multiple_networks(num_networks=3)
    print(f"Generated {len(multiple_results)} networks")