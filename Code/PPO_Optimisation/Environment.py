
"""

In this function, we autogenerate an initial water network environment. We define the conditions for how that environment updates with each time step, given a scenario input

"""

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import copy

# Define network component types
class ComponentType(Enum):
    RESERVOIR = 0
    PUMP = 1
    TANK = 2
    JUNCTION = 3
    PIPE = 4

# Define network topology types
class NetworkTopology(Enum):
    BRANCHED = 0
    LOOPED = 1
    SEMI_LOOPED = 2

@dataclass
class Component:
    id: str
    type: ComponentType
    position: Tuple[float, float]  # x, y coordinates

@dataclass
class Reservoir(Component):
    head: float  # hydraulic head (m)
    
@dataclass
class Pump(Component):
    flow_rate: float  # max flow rate (L/s)
    power: float  # power (kW)
    efficiency: float  # efficiency (0-1)
    
@dataclass
class Tank(Component):
    volume: float  # max volume (m³)
    level: float  # current water level (m)
    diameter: float  # diameter (m)
    min_level: float  # minimum level (m)
    max_level: float  # maximum level (m)
    
@dataclass
class Junction(Component):
    demand: float  # water demand (L/s)
    elevation: float  # elevation (m)
    pressure: float = 0.0  # pressure (m)

@dataclass
class Pipe:
    id: str
    start_node: str  # ID of starting component
    end_node: str  # ID of ending component
    length: float  # length (m)
    diameter: float  # diameter (mm)
    roughness: float  # roughness coefficient
    flow: float = 0.0  # flow (L/s)
    velocity: float = 0.0  # velocity (m/s)
    headloss: float = 0.0  # head loss (m)

@dataclass
class WaterNetwork:
    components: Dict[str, Component] = field(default_factory=dict)
    pipes: Dict[str, Pipe] = field(default_factory=dict)
    topology: NetworkTopology = NetworkTopology.LOOPED
    graph: nx.Graph = field(default_factory=nx.Graph)
    
    def add_component(self, component: Component):
        self.components[component.id] = component
        self.graph.add_node(component.id, 
                           type=component.type,
                           pos=component.position)
    
    def add_pipe(self, pipe: Pipe):
        self.pipes[pipe.id] = pipe
        self.graph.add_edge(pipe.start_node, pipe.end_node, 
                           id=pipe.id,
                           length=pipe.length,
                           diameter=pipe.diameter,
                           roughness=pipe.roughness)
    
    def get_component(self, component_id: str) -> Optional[Component]:
        return self.components.get(component_id)
    
    def get_pipe(self, pipe_id: str) -> Optional[Pipe]:
        return self.pipes.get(pipe_id)
    
    def get_all_components_by_type(self, component_type: ComponentType) -> List[Component]:
        return [c for c in self.components.values() if c.type == component_type]
    
    def get_connected_pipes(self, component_id: str) -> List[Pipe]:
        connected_pipes = []
        for pipe in self.pipes.values():
            if pipe.start_node == component_id or pipe.end_node == component_id:
                connected_pipes.append(pipe)
        return connected_pipes
    
    def visualise(self, title: str = "Water Distribution Network", modifications: Optional[Dict[str, float]] = None):
        plt.figure(figsize=(12, 8))
        
        # Define colors and shapes for different component types
        colors = {
            ComponentType.RESERVOIR: 'blue',
            ComponentType.PUMP: 'red',
            ComponentType.TANK: 'green',
            ComponentType.JUNCTION: 'gray',
            # If componennt type is pipe and in modifications, colour = 'orange', otherwise 'black'
            ComponentType.PIPE: 'orange' if modifications else 'black'
        }
        
        shapes = {
            ComponentType.RESERVOIR: 's',  # square
            ComponentType.PUMP: '^',       # triangle up
            ComponentType.TANK: 'o',       # circle
            ComponentType.JUNCTION: '.'    # point
        }
        
        # Get positions for all nodes
        pos = {node: self.components[node].position for node in self.graph.nodes()}
        
        # Draw nodes by type
        for comp_type in ComponentType:
            if comp_type != ComponentType.PIPE:
                node_list = [n for n in self.graph.nodes() if self.components[n].type == comp_type]
                if node_list:
                    nx.draw_networkx_nodes(
                        self.graph, pos,
                        nodelist=node_list,
                        node_color=colors[comp_type],
                        node_shape=shapes[comp_type],
                        node_size=300 if comp_type != ComponentType.JUNCTION else 100
                    )

        # Draw edges with width based on pipe diameter
        edge_list = []
        edge_widths = []
        edge_colors = []

        for pipe_id, pipe in self.pipes.items():
            edge_list.append((pipe.start_node, pipe.end_node))
            edge_widths.append(1 + pipe.diameter / 100)  # Scale the width
            edge_colors.append('orange' if modifications and pipe_id in modifications else 'black')

        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=edge_list,
            width=edge_widths,
            edge_color=edge_colors
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker=shapes[comp_type], color='w', label=comp_type.name,
                          markerfacecolor=colors[comp_type], markersize=10) for comp_type in ComponentType if comp_type != ComponentType.PIPE
        ]
        
        plt.title(title)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def generate_initial_network(
    network_size: int,
    topology: NetworkTopology,
    available_diameters: List[float],
    available_roughness: List[float],
    area_size: Tuple[float, float] = (1000, 1000),
    num_reservoirs: int = 1,
    num_pumps: int = 1,
    num_tanks: int = 1,
    min_elevation: float = 0.0,
    max_elevation: float = 100.0,
    min_demand: float = 0.1,
    max_demand: float = 5.0,
    min_pipe_length: float = 100.0,
    max_pipe_length: float = 1000.0
) -> WaterNetwork:
    """
    Generate a realistic initial water distribution network.
    
    Args:
        network_size: Number of junctions in the network
        topology: Type of network topology (branched, looped, semi-looped)
        available_diameters: List of available pipe diameters (mm)
        available_roughness: List of available pipe roughness values
        area_size: Size of the area (width, height) in meters
        num_reservoirs: Number of reservoirs in the network
        num_pumps: Number of pumps in the network
        num_tanks: Number of tanks in the network
        min_elevation: Minimum elevation for junctions (m)
        max_elevation: Maximum elevation for junctions (m)
        min_demand: Minimum water demand at junctions (L/s)
        max_demand: Maximum water demand at junctions (L/s)
        min_pipe_length: Minimum pipe length (m)
        max_pipe_length: Maximum pipe length (m)
        
    Returns:
        WaterNetwork: A complete water distribution network
    """
    network = WaterNetwork(topology=topology)
    
    # Step 1: Create reservoirs at elevated positions
    for i in range(num_reservoirs):
        reservoir_id = f"R{i+1}"
        # Place reservoirs at elevated positions
        x = random.uniform(0, area_size[0])
        y = random.uniform(0, area_size[1])
        head = max_elevation * 1.2  # Higher than max elevation
        reservoir = Reservoir(
            id=reservoir_id,
            type=ComponentType.RESERVOIR,
            position=(x, y),
            head=head
        )
        network.add_component(reservoir)
    
    # Step 2: Create pumps
    for i in range(num_pumps):
        pump_id = f"P{i+1}"
        # Place pumps near reservoirs
        reservoir = network.get_all_components_by_type(ComponentType.RESERVOIR)[i % num_reservoirs]
        rx, ry = reservoir.position
        x = rx + random.uniform(-100, 100)
        y = ry + random.uniform(-100, 100)
        
        pump = Pump(
            id=pump_id,
            type=ComponentType.PUMP,
            position=(x, y),
            flow_rate=random.uniform(10, 50),  # L/s
            power=random.uniform(5, 20),       # kW
            efficiency=random.uniform(0.7, 0.9)
        )
        network.add_component(pump)
    
    # Step 3: Create tanks
    for i in range(num_tanks):
        tank_id = f"T{i+1}"
        # Place tanks at strategic locations
        x = random.uniform(0, area_size[0])
        y = random.uniform(0, area_size[1])
        
        diameter = random.uniform(5, 15)  # m
        max_level = random.uniform(3, 10)  # m
        min_level = random.uniform(0.5, 1.5)  # m
        tank = Tank(
            id=tank_id,
            type=ComponentType.TANK,
            position=(x, y),
            volume=np.pi * (diameter/2)**2 * max_level,  # m³
            level=random.uniform(min_level, max_level),  # m
            diameter=diameter,
            min_level=min_level,
            max_level=max_level
        )
        network.add_component(tank)
    
    # Step 4: Create junctions with demands
    for i in range(network_size):
        junction_id = f"J{i+1}"
        x = random.uniform(0, area_size[0])
        y = random.uniform(0, area_size[1])
        
        junction = Junction(
            id=junction_id,
            type=ComponentType.JUNCTION,
            position=(x, y),
            demand=random.uniform(min_demand, max_demand),
            elevation=random.uniform(min_elevation, max_elevation)
        )
        network.add_component(junction)
    
    # Step 5: Connect components with pipes based on topology
    if topology == NetworkTopology.BRANCHED:
        _create_branched_topology(network, available_diameters, available_roughness, 
                                 min_pipe_length, max_pipe_length)
    elif topology == NetworkTopology.LOOPED:
        _create_looped_topology(network, available_diameters, available_roughness, 
                               min_pipe_length, max_pipe_length)
    else:  # SEMI_LOOPED
        _create_semi_looped_topology(network, available_diameters, available_roughness, 
                                    min_pipe_length, max_pipe_length)
    
    return network

def _create_branched_topology(
    network: WaterNetwork,
    available_diameters: List[float],
    available_roughness: List[float],
    min_pipe_length: float,
    max_pipe_length: float
):
    """Create a branched (tree-like) network topology."""
    # Get all components
    all_components = list(network.components.keys())
    connected = set()
    
    # Start with source nodes (reservoirs and pumps)
    sources = [c.id for c in network.components.values() 
               if c.type in (ComponentType.RESERVOIR, ComponentType.PUMP)]
    
    # Ensure we have at least one source
    if not sources:
        return
    
    connected.update(sources)
    pipe_id = 1
    
    # Create a minimum spanning tree
    while len(connected) < len(all_components):
        # Find closest unconnected node to any connected node
        min_dist = float('inf')
        closest_pair = None
        
        for connected_id in connected:
            connected_pos = network.components[connected_id].position
            
            for component_id in all_components:
                if component_id not in connected:
                    component_pos = network.components[component_id].position
                    
                    # Calculate Euclidean distance
                    dist = np.sqrt((connected_pos[0] - component_pos[0])**2 + 
                                  (connected_pos[1] - component_pos[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (connected_id, component_id)
        
        if closest_pair:
            start_node, end_node = closest_pair
            
            # Ensure the pipe length is within bounds
            pipe_length = max(min_pipe_length, min(max_pipe_length, min_dist))
            
            # Select random diameter and roughness
            diameter = random.choice(available_diameters)
            roughness = random.choice(available_roughness)
            
            pipe = Pipe(
                id=f"PIPE{pipe_id}",
                start_node=start_node,
                end_node=end_node,
                length=pipe_length,
                diameter=diameter,
                roughness=roughness
            )
            
            network.add_pipe(pipe)
            connected.add(end_node)
            pipe_id += 1

def _create_looped_topology(
    network: WaterNetwork,
    available_diameters: List[float],
    available_roughness: List[float],
    min_pipe_length: float,
    max_pipe_length: float
):
    """Create a looped network topology with redundant paths."""
    # First create a branched topology as a starting point
    _create_branched_topology(network, available_diameters, available_roughness,
                             min_pipe_length, max_pipe_length)
    
    # Add additional pipes to create loops
    # We'll add approximately 40% more pipes to create loops
    num_components = len(network.components)
    num_existing_pipes = len(network.pipes)
    num_additional_pipes = int(num_existing_pipes * 0.4)
    pipe_id = num_existing_pipes + 1
    
    # Collect node positions for distance calculations
    node_positions = {node_id: comp.position for node_id, comp in network.components.items()}
    
    # Add loops by connecting nearby nodes that aren't already connected
    added_pipes = 0
    max_attempts = num_additional_pipes * 5  # Limit attempts to avoid infinite loops
    attempts = 0
    
    while added_pipes < num_additional_pipes and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two nodes that aren't source nodes and aren't already connected
        nodes = list(network.components.keys())
        
        # Filter out nodes that are directly connected
        candidate_pairs = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                
                # Skip if they're already connected
                if network.graph.has_edge(node1, node2):
                    continue
                    
                # Calculate distance
                pos1, pos2 = node_positions[node1], node_positions[node2]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Only consider if distance is reasonable
                if min_pipe_length <= dist <= max_pipe_length * 1.5:
                    candidate_pairs.append((node1, node2, dist))
        
        if not candidate_pairs:
            continue
            
        # Select a pair with preference to shorter distances
        candidate_pairs.sort(key=lambda x: x[2])
        selected_pair = candidate_pairs[0] if len(candidate_pairs) < 3 else random.choice(candidate_pairs[:3])
        
        start_node, end_node, dist = selected_pair
        
        # Select random diameter and roughness
        diameter = random.choice(available_diameters)
        roughness = random.choice(available_roughness)
        
        pipe = Pipe(
            id=f"PIPE{pipe_id}",
            start_node=start_node,
            end_node=end_node,
            length=dist,
            diameter=diameter,
            roughness=roughness
        )
        
        network.add_pipe(pipe)
        added_pipes += 1
        pipe_id += 1

def _create_semi_looped_topology(
    network: WaterNetwork,
    available_diameters: List[float],
    available_roughness: List[float],
    min_pipe_length: float,
    max_pipe_length: float
):
    """Create a semi-looped network topology (mix of branched and looped sections)."""
    # First create a branched topology as a starting point
    _create_branched_topology(network, available_diameters, available_roughness,
                             min_pipe_length, max_pipe_length)
    
    # Add additional pipes to create some loops, but fewer than in fully looped topology
    # We'll add approximately 20% more pipes to create partial loops
    num_components = len(network.components)
    num_existing_pipes = len(network.pipes)
    num_additional_pipes = int(num_existing_pipes * 0.2)
    
    # Use the same logic as looped topology but with fewer additional pipes
    pipe_id = num_existing_pipes + 1
    
    # Collect node positions for distance calculations
    node_positions = {node_id: comp.position for node_id, comp in network.components.items()}
    
    # Add loops by connecting nearby nodes that aren't already connected
    added_pipes = 0
    max_attempts = num_additional_pipes * 5  # Limit attempts to avoid infinite loops
    attempts = 0
    
    # Focus on creating loops in high-demand areas
    junctions = {j.id: j for j in network.get_all_components_by_type(ComponentType.JUNCTION)}
    high_demand_junctions = sorted(junctions.items(), key=lambda x: x[1].demand, reverse=True)
    high_demand_junction_ids = [j[0] for j in high_demand_junctions[:int(len(high_demand_junctions)/3)]]
    
    while added_pipes < num_additional_pipes and attempts < max_attempts:
        attempts += 1
        
        # Prioritize connecting high-demand junctions
        if high_demand_junction_ids and random.random() < 0.7:
            node1 = random.choice(high_demand_junction_ids)
            # Find a nearby junction
            candidates = []
            for node2, comp in network.components.items():
                if node1 != node2 and not network.graph.has_edge(node1, node2):
                    pos1, pos2 = node_positions[node1], node_positions[node2]
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if min_pipe_length <= dist <= max_pipe_length * 1.2:
                        candidates.append((node2, dist))
            
            if candidates:
                candidates.sort(key=lambda x: x[1])
                node2 = candidates[0][0] if len(candidates) < 3 else random.choice(candidates[:3])[0]
                dist = np.sqrt(sum((np.array(node_positions[node1]) - np.array(node_positions[node2]))**2))
                
                # Select random diameter and roughness
                diameter = random.choice(available_diameters)
                roughness = random.choice(available_roughness)
                
                pipe = Pipe(
                    id=f"PIPE{pipe_id}",
                    start_node=node1,
                    end_node=node2,
                    length=dist,
                    diameter=diameter,
                    roughness=roughness
                )
                
                network.add_pipe(pipe)
                added_pipes += 1
                pipe_id += 1
        else:
            # Regular random connections like in looped topology
            nodes = list(network.components.keys())
            node1, node2 = random.sample(nodes, 2)
            
            # Skip if they're already connected
            if network.graph.has_edge(node1, node2):
                continue
                
            # Calculate distance
            pos1, pos2 = node_positions[node1], node_positions[node2]
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            # Only consider if distance is reasonable
            if not (min_pipe_length <= dist <= max_pipe_length * 1.5):
                continue
                
            # Select random diameter and roughness
            diameter = random.choice(available_diameters)
            roughness = random.choice(available_roughness)
            
            pipe = Pipe(
                id=f"PIPE{pipe_id}",
                start_node=node1,
                end_node=node2,
                length=dist,
                diameter=diameter,
                roughness=roughness
            )
            
            network.add_pipe(pipe)
            added_pipes += 1
            pipe_id += 1

def update_network(
    network: WaterNetwork,
    selected_action: Dict[str, float],  # Pipe ID -> new diameter
    updated_demands: Dict[str, float],  # Junction ID -> new demand
    updated_supply: Dict[str, float]    # Source ID (reservoir/pump) -> new supply
) -> WaterNetwork:
    """
    Update the network based on selected actions and new supply/demand values.
    
    Args:
        network: The current water distribution network
        selected_action: Dictionary mapping pipe IDs to new diameters
        updated_demands: Dictionary mapping junction IDs to new demand values
        updated_supply: Dictionary mapping source IDs to new supply values
        
    Returns:
        WaterNetwork: The updated network
    """
    # Make a deep copy of the network to avoid modifying the original
    updated_network = copy.deepcopy(network)
    
    # Step 1: Update pipe diameters based on selected actions
    for pipe_id, new_diameter in selected_action.items():
        if pipe_id in updated_network.pipes:
            pipe = updated_network.pipes[pipe_id]
            old_diameter = pipe.diameter
            pipe.diameter = new_diameter
            
            # Update the graph edge attribute
            updated_network.graph[pipe.start_node][pipe.end_node]['diameter'] = new_diameter
    
    # Step 2: Update junction demands
    for junction_id, new_demand in updated_demands.items():
        component = updated_network.get_component(junction_id)
        if component and component.type == ComponentType.JUNCTION:
            component.demand = new_demand
    
    # Step 3: Update reservoir/pump supply
    for source_id, new_supply in updated_supply.items():
        component = updated_network.get_component(source_id)
        if component:
            if component.type == ComponentType.RESERVOIR:
                # For reservoirs, supply might be represented as head
                component.head = new_supply
            elif component.type == ComponentType.PUMP:
                # For pumps, supply is the flow rate
                component.flow_rate = new_supply
    
    return updated_network

def generate_demand_scenarios(
    network: WaterNetwork,
    num_scenarios: int,
    demand_variation: float = 0.2,  # Percentage variation around base demand
    time_patterns: Optional[Dict[str, List[float]]] = None  # Time-based patterns
) -> List[Dict[str, float]]:
    """
    Generate multiple demand scenarios based on probabilistic distributions.
    
    Args:
        network: The water distribution network
        num_scenarios: Number of demand scenarios to generate
        demand_variation: Variation coefficient around base demand
        time_patterns: Optional dictionary of time patterns (e.g., daily, weekly)
        
    Returns:
        List of demand scenario dictionaries (Junction ID -> demand)
    """
    scenarios = []
    junctions = [c for c in network.components.values() if c.type == ComponentType.JUNCTION]
    
    # If no time patterns provided, create a simple daily pattern
    if not time_patterns:
        time_patterns = {
            "residential": [0.6, 0.5, 0.4, 0.4, 0.5, 0.7, 0.9, 1.2, 1.3, 1.2, 1.1, 1.0,
                          1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.7],
            "commercial": [0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.4, 1.5, 1.5,
                         1.4, 1.5, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5],
            "industrial": [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.2, 1.2, 1.2,
                         1.2, 1.2, 1.2, 1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
        }
    
    # Assign a random user type to each junction for demonstration
    user_types = ["residential", "commercial", "industrial"]
    junction_types = {j.id: random.choice(user_types) for j in junctions}
    
    # Generate scenarios
    for scenario_idx in range(num_scenarios):
        # Select a random hour of the day for this scenario
        hour = random.randint(0, 23)
        
        scenario_demands = {}
        for junction in junctions:
            base_demand = junction.demand
            user_type = junction_types[junction.id]
            time_factor = time_patterns[user_type][hour]
            
            # Apply time pattern and add random variation
            variation = random.uniform(1 - demand_variation, 1 + demand_variation)
            new_demand = base_demand * time_factor * variation
            
            scenario_demands[junction.id] = max(0.0, new_demand)  # Ensure non-negative demand
        
        scenarios.append(scenario_demands)
    
    return scenarios

def generate_supply_scenarios(
    network: WaterNetwork,
    num_scenarios: int,
    supply_variation: float = 0.1  # Percentage variation around base supply
) -> List[Dict[str, float]]:
    """
    Generate multiple supply scenarios based on probabilistic distributions.
    
    Args:
        network: The water distribution network
        num_scenarios: Number of supply scenarios to generate
        supply_variation: Variation coefficient around base supply
        
    Returns:
        List of supply scenario dictionaries (Source ID -> supply)
    """
    scenarios = []
    
    # Get all supply sources
    reservoirs = [c for c in network.components.values() if c.type == ComponentType.RESERVOIR]
    pumps = [c for c in network.components.values() if c.type == ComponentType.PUMP]
    
    for scenario_idx in range(num_scenarios):
        scenario_supply = {}
        
        # Reservoir supply variations (head)
        for reservoir in reservoirs:
            base_head = reservoir.head
            variation = random.uniform(1 - supply_variation, 1 + supply_variation)
            scenario_supply[reservoir.id] = base_head * variation
        
        # Pump supply variations (flow rate)
        for pump in pumps:
            base_flow = pump.flow_rate
            variation = random.uniform(1 - supply_variation, 1 + supply_variation)
            scenario_supply[pump.id] = base_flow * variation
        
        scenarios.append(scenario_supply)
    
    return scenarios

# Example usage:
if __name__ == "__main__":
    # Available pipe configurations
    available_diameters = [100, 150, 200, 250, 300, 400, 500]  # mm
    available_roughness = [0.01, 0.015, 0.02, 0.025, 0.03]  # Darcy-Weisbach coefficients
    
    # Generate a sample network
    network = generate_initial_network(
        network_size=20,
        topology=NetworkTopology.LOOPED,
        available_diameters=available_diameters,
        available_roughness=available_roughness,
        num_reservoirs=1,
        num_pumps=2,
        num_tanks=2
    )
    
    # visualise the network
    network.visualise("Initial Water Distribution Network")
    
    # Generate demand scenarios
    demand_scenarios = generate_demand_scenarios(network, num_scenarios=10)
    
    # Generate supply scenarios
    supply_scenarios = generate_supply_scenarios(network, num_scenarios=10)
    
    # Example action: change some pipe diameters
    sample_action = {f"PIPE{i}": random.choice(available_diameters) for i in range(1, 6)}
    
    # Update the network
    updated_network = update_network(
        network=network,
        selected_action=sample_action,
        updated_demands=demand_scenarios[0],
        updated_supply=supply_scenarios[0]
    )

    print(f"Sample action: {sample_action}")
    
    # visualise the updated network
    updated_network.visualise("Updated Water Distribution Network", modifications=sample_action)
    
    print(f"Generated {len(demand_scenarios)} demand scenarios and {len(supply_scenarios)} supply scenarios")
    print(f"Updated {len(sample_action)} pipes in the network")