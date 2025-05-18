
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
import os

class HydraulicNetwork:
    def __init__(self, network_type='looped', num_junctions=10, grid_size=1000, seed=None):
        """
        Initialize a hydraulic network generator
        
        Args:
            network_type (str): 'looped' or 'branched'
            num_junctions (int): Number of junctions in the network
            grid_size (int): Size of the grid (meters)
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.network_type = network_type
        self.num_junctions = num_junctions
        self.grid_size = grid_size
        self.G = nx.Graph()
        self.node_positions = {}
        self.elevation_map = None
        self.junction_demand_patterns = {}
        self.pipe_diameters = {}
        
        # Node types
        self.reservoirs = []
        self.tanks = []
        self.junctions = []
        self.pumps = []
        
    def generate_elevation_map(self, peak_height=100):
        """Generate a simple contour map for elevation data"""
        # Create a grid for the elevation map
        x = np.linspace(0, self.grid_size, 100)
        y = np.linspace(0, self.grid_size, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create a simple hill-like elevation
        sigma_x = self.grid_size / 4
        sigma_y = self.grid_size / 4
        
        # Create two peaks
        peak1_x = self.grid_size * 0.7  # First peak location
        peak1_y = self.grid_size * 0.7
        peak2_x = self.grid_size * 0.3  # Second peak location
        peak2_y = self.grid_size * 0.3
        
        # Calculate heights based on distance from peaks
        Z1 = peak_height * np.exp(-((X - peak1_x)**2 / (2 * sigma_x**2) + (Y - peak1_y)**2 / (2 * sigma_y**2)))
        Z2 = peak_height * 0.8 * np.exp(-((X - peak2_x)**2 / (2 * sigma_x**2) + (Y - peak2_y)**2 / (2 * sigma_y**2)))
        
        # Combine peaks and add a base elevation
        Z = Z1 + Z2 + 10  # Base elevation of 10 meters
        
        self.elevation_map = {
            'X': X,
            'Y': Y,
            'Z': Z,
            'peak1': (peak1_x, peak1_y),
            'peak2': (peak2_x, peak2_y)
        }
        
        return self.elevation_map
    
    def get_elevation(self, x, y):
        """Get elevation at specific coordinates by interpolating from the elevation map"""
        if self.elevation_map is None:
            self.generate_elevation_map()
            
        # Find nearest points in the grid
        X = self.elevation_map['X']
        Y = self.elevation_map['Y']
        Z = self.elevation_map['Z']
        
        # Simple bilinear interpolation
        x_idx = np.abs(X[0, :] - x).argmin()
        y_idx = np.abs(Y[:, 0] - y).argmin()
        
        return Z[y_idx, x_idx]
    
    def generate_demand_pattern(self, base_demand=1.0):
        """Generate a 24-hour demand pattern with morning and evening peaks"""
        # Create a 24-hour demand pattern
        hours = np.arange(24)
        
        # Base pattern with morning and evening peaks
        pattern = 0.5 + 0.5 * np.sin(np.pi * (hours - 6) / 12)
        pattern[hours < 5] = 0.3  # Low demand at night
        pattern[hours > 22] = 0.3  # Low demand late night
        
        # Add some noise
        pattern = pattern + 0.1 * np.random.normal(0, 1, 24)
        pattern = np.clip(pattern, 0.3, 1.5)  # Keep within reasonable bounds
        
        # Normalize to average to base_demand
        pattern = pattern * base_demand / np.mean(pattern)
        
        return pattern.tolist()
    
    def generate_head_pattern(self, base_head=100.0):
        """Generate a 24-hour head pattern for reservoirs"""
        # Create a 24-hour head pattern
        hours = np.arange(24)
        
        # Base pattern with slight fluctuations
        pattern = base_head + 2 * np.sin(np.pi * hours / 12)
        
        # Add some noise
        pattern = pattern + 0.5 * np.random.normal(0, 1, 24)
        
        return pattern.tolist()
    
    def generate_looped_network(self):
        """Generate a looped network with one reservoir, one tank, and one pump"""
        # Place junctions in a grid-like pattern
        junctions_per_side = int(np.ceil(np.sqrt(self.num_junctions)))
        spacing = self.grid_size / (junctions_per_side + 1)
        
        # Create junctions
        for i in range(self.num_junctions):
            row = i // junctions_per_side + 1
            col = i % junctions_per_side + 1
            
            # Add some randomness to the grid positions
            x = col * spacing + random.uniform(-spacing/4, spacing/4)
            y = row * spacing + random.uniform(-spacing/4, spacing/4)
            
            junction_id = f"J{i+1}"
            self.junctions.append(junction_id)
            elevation = self.get_elevation(x, y)
            self.G.add_node(junction_id, 
                           type='junction', 
                           elevation=elevation, 
                           demand_pattern=self.generate_demand_pattern(
                               base_demand=random.uniform(0.5, 2.0)
                           ))
            self.node_positions[junction_id] = (x, y)
            
        # Create looped connections between junctions
        for i in range(self.num_junctions):
            # Connect to right neighbor in grid
            if (i+1) % junctions_per_side != 0 and i+1 < self.num_junctions:
                self.G.add_edge(f"J{i+1}", f"J{i+2}")
                self.pipe_diameters[(f"J{i+1}", f"J{i+2}")] = random.choice([100, 150, 200])
            
            # Connect to bottom neighbor in grid
            if i + junctions_per_side < self.num_junctions:
                self.G.add_edge(f"J{i+1}", f"J{i+junctions_per_side+1}")
                self.pipe_diameters[(f"J{i+1}", f"J{i+junctions_per_side+1}")] = random.choice([100, 150, 200])
        
        # Add reservoir at one of the highest points
        peak1 = self.elevation_map['peak1']
        reservoir_id = "R1"
        self.reservoirs.append(reservoir_id)
        base_head = self.get_elevation(peak1[0], peak1[1]) + 20  # Head above elevation
        self.G.add_node(reservoir_id, 
                       type='reservoir', 
                       base_head=base_head,
                       head_pattern=self.generate_head_pattern(base_head))
        self.node_positions[reservoir_id] = peak1
        
        # Connect reservoir to nearest junction
        nearest_junction = self.find_nearest_node(peak1, node_type='junction')
        self.G.add_edge(reservoir_id, nearest_junction)
        self.pipe_diameters[(reservoir_id, nearest_junction)] = 300  # Large diameter for reservoir connection
        
        # Add tank at another high point
        peak2 = self.elevation_map['peak2']
        tank_id = "T1"
        self.tanks.append(tank_id)
        elevation = self.get_elevation(peak2[0], peak2[1])
        self.G.add_node(tank_id, 
                       type='tank',
                       elevation=elevation,
                       initial_level=10,
                       min_level=0,
                       max_level=20,
                       diameter=15)
        self.node_positions[tank_id] = peak2
        
        # Connect tank to nearest junction
        nearest_junction = self.find_nearest_node(peak2, node_type='junction')
        self.G.add_edge(tank_id, nearest_junction)
        self.pipe_diameters[(tank_id, nearest_junction)] = 200  # Medium diameter for tank connection
        
        # Add a pump
        # Find a connection where we can place the pump
        # Ideal is from a low to a higher area
        for edge in self.G.edges():
            node1, node2 = edge
            if (self.G.nodes[node1].get('type') == 'junction' and 
                self.G.nodes[node2].get('type') == 'junction'):
                
                elev1 = self.G.nodes[node1]['elevation']
                elev2 = self.G.nodes[node2]['elevation']
                
                # If we have a significant elevation difference, place pump
                if abs(elev1 - elev2) > 5:
                    pump_id = "P1"
                    self.pumps.append(pump_id)
                    # Remove direct connection
                    self.G.remove_edge(node1, node2)
                    # Add pump and connect
                    self.G.add_node(pump_id, type='pump', power=20)
                    self.G.add_edge(node1, pump_id)
                    self.G.add_edge(pump_id, node2)
                    
                    # Place pump between the junctions
                    pos1 = self.node_positions[node1]
                    pos2 = self.node_positions[node2]
                    self.node_positions[pump_id] = ((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2)
                    
                    self.pipe_diameters[(node1, pump_id)] = 200
                    self.pipe_diameters[(pump_id, node2)] = 200
                    break
                    
        # If no pump was added due to lack of elevation difference, add one anyway
        if not self.pumps:
            # Find a central junction
            central_junction = self.junctions[len(self.junctions) // 2]
            # Find a neighbor
            neighbors = list(self.G.neighbors(central_junction))
            if neighbors:
                neighbor = neighbors[0]
                if self.G.nodes[neighbor].get('type') == 'junction':
                    pump_id = "P1"
                    self.pumps.append(pump_id)
                    # Remove direct connection
                    self.G.remove_edge(central_junction, neighbor)
                    # Add pump and connect
                    self.G.add_node(pump_id, type='pump', power=15)
                    self.G.add_edge(central_junction, pump_id)
                    self.G.add_edge(pump_id, neighbor)
                    
                    # Place pump between the junctions
                    pos1 = self.node_positions[central_junction]
                    pos2 = self.node_positions[neighbor]
                    self.node_positions[pump_id] = ((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2)
                    
                    self.pipe_diameters[(central_junction, pump_id)] = 200
                    self.pipe_diameters[(pump_id, neighbor)] = 200
                    
        return self.G, self.node_positions
    
    def generate_branched_network(self):
        """Generate a branched network with a main pipe, secondary pipes, and a tank."""
        
        # Add reservoir at one of the highest points
        peak1 = self.elevation_map['peak1']
        reservoir_id = "R1"
        self.reservoirs.append(reservoir_id)
        base_head = self.get_elevation(peak1[0], peak1[1]) + 20  # Head above elevation
        self.G.add_node(reservoir_id, 
                       type='reservoir', 
                       base_head=base_head,
                       head_pattern=self.generate_head_pattern(base_head))
        self.node_positions[reservoir_id] = peak1

        peak2 = self.elevation_map['peak2']
        tank_id = "T1"
        self.tanks.append(tank_id)
        elevation = self.get_elevation(peak2[0], peak2[1])
        self.G.add_node(tank_id,
                          type='tank',
                          elevation=elevation,
                          initial_level=10,
                          min_level=0,
                          max_level=20,
                          diameter=15)
        self.node_positions[tank_id] = peak2

        # Create junctions along the line from the reservoir to the tank
        num_junctions = self.num_junctions
        for i in range(int(num_junctions/2)): # Only half junctions along main pipe
            # Calculate position along the line
            x = peak1[0] + (peak2[0] - peak1[0]) * (i + 1) / (num_junctions/2 + 1)
            y = peak1[1] + (peak2[1] - peak1[1]) * (i + 1) / (num_junctions/2 + 1)
            junction_id = f"J{i+1}"
            self.junctions.append(junction_id)
            elevation = self.get_elevation(x, y)
            self.G.add_node(junction_id,
                            type='junction', 
                            elevation=elevation, 
                            demand_pattern=self.generate_demand_pattern(
                                 base_demand=random.uniform(0.5, 2.0)
                            ))
            self.node_positions[junction_id] = (x, y)

        # Connect reservoir to the first junction and so on with large diameter pipe
        current_node = reservoir_id
        for i in range(int(num_junctions/2)):
            next_node = f"J{i+1}"
            self.G.add_edge(current_node, next_node)
            self.pipe_diameters[(current_node, next_node)] = 300
            current_node = next_node

        # Connect the last junction to the tank with a large diameter pipe
        self.G.add_edge(current_node, tank_id)
        self.pipe_diameters[(current_node, tank_id)] = 300

                # Create secondary branches from the main pipe
        junction_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'junction']
        
        # Track which nodes are on the main pipe vs secondary branches
        main_pipe_nodes = [f"J{i+1}" for i in range(int(num_junctions/2))]
        secondary_branch_nodes = []
        
        # First create secondary branches
        for i in range(int(num_junctions/2), int(num_junctions*3/4)):
            # Filter for main pipe junction nodes with less than 3 connections
            available_nodes = [n for n in main_pipe_nodes if self.G.degree(n) < 3]
            if not available_nodes:
                break  # No suitable nodes available
                
            main_node = random.choice(available_nodes)
            
            # Calculate position for the new junction - it should be perpendicular to the main pipe
            pos = self.node_positions[main_node]

            main_pipe_neighbors = [n for n in self.G.neighbors(main_node) 
                                  if self.G.nodes[n].get('type') in ['junction', 'reservoir', 'tank']]
            
            if main_pipe_neighbors:
                # Get position of a neighbor to determine main pipe direction
                neighbor_pos = self.node_positions[main_pipe_neighbors[0]]
                
                # Calculate the direction vector of the main pipe
                dx = neighbor_pos[0] - pos[0]
                dy = neighbor_pos[1] - pos[1]
                
                # Calculate perpendicular direction (rotate 90 degrees)
                perp_dx = -dy
                perp_dy = dx
                
                # Normalize the perpendicular vector
                length = (perp_dx**2 + perp_dy**2)**0.5
                if length > 0:
                    perp_dx /= length
                    perp_dy /= length
                
                # Set the new junction's position perpendicular to the main pipe
                # with a random distance between 50 and 200 meters
                distance = random.uniform(50, 200)
                # Randomly choose side (left or right perpendicular)
                side = random.choice([-1, 1])
                
                x = pos[0] + side * perp_dx * distance
                y = pos[1] + side * perp_dy * distance
            else:
                # Fallback if no neighbors found
                angle = random.uniform(0, 2*np.pi)
                distance = random.uniform(self.grid_size/5, self.grid_size/2)
                x = pos[0] + distance * np.cos(angle)
                y = pos[1] + distance * np.sin(angle)

            junction_id = f"J{i+1}"
            self.junctions.append(junction_id)
            secondary_branch_nodes.append(junction_id)  # Track as a secondary branch node
            elevation = self.get_elevation(x, y)
            
            self.G.add_node(junction_id,
                            type='junction', 
                            elevation=elevation, 
                            demand_pattern=self.generate_demand_pattern(
                                base_demand=random.uniform(0.5, 2.0)
                            ))
            self.node_positions[junction_id] = (x, y)
            
            # Connect the new junction to the main node
            self.G.add_edge(main_node, junction_id)
            self.pipe_diameters[(main_node, junction_id)] = random.choice([100, 150])  # Smaller diameters for branches
        
        # Now create tertiary branches from secondary branches
        for i in range(int(num_junctions*3/4), num_junctions):
            # Filter for secondary branch junction nodes with less than 3 connections
            available_nodes = [n for n in secondary_branch_nodes if self.G.degree(n) < 2]
            if not available_nodes:
                break  # No suitable nodes available
                
            branch_node = random.choice(available_nodes)
            
            # Calculate position for the new junction
            pos = self.node_positions[branch_node]
            
            # Get the angle of the existing branch
            neighbor = list(self.G.neighbors(branch_node))[0]  # Secondary branch has only one neighbor
            neighbor_pos = self.node_positions[neighbor]
            
            # Calculate the direction vector of the secondary branch
            dx = pos[0] - neighbor_pos[0]
            dy = pos[1] - neighbor_pos[1]
            
            # Normalize the vector
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx /= length
                dy /= length
            
            # Add some randomness to the angle (+-45 degrees)
            angle_variation = random.uniform(-np.pi/4, np.pi/4)
            new_dx = dx*np.cos(angle_variation) - dy*np.sin(angle_variation)
            new_dy = dx*np.sin(angle_variation) + dy*np.cos(angle_variation)
            
            # Set the new junction's position extending from the secondary branch
            distance = random.uniform(50, 150)  # Shorter tertiary branches
            x = pos[0] + new_dx * distance
            y = pos[1] + new_dy * distance

            junction_id = f"J{i+1}"
            self.junctions.append(junction_id)
            elevation = self.get_elevation(x, y)
            
            self.G.add_node(junction_id,
                            type='junction', 
                            elevation=elevation, 
                            demand_pattern=self.generate_demand_pattern(
                                base_demand=random.uniform(0.3, 1.5)  # Smaller demands for tertiary branches
                            ))
            self.node_positions[junction_id] = (x, y)
            
            # Connect the new junction to the secondary branch node
            self.G.add_edge(branch_node, junction_id)
            self.pipe_diameters[(branch_node, junction_id)] = random.choice([50, 100])  # Even smaller diameters for tertiary branches
        
        # Add a pump on the main pipe
        # Find a suitable connection on the main pipe
        main_pipe_edges = []
        # First collect the main pipe edges
        current_node = reservoir_id
        for i in range(int(num_junctions/2)):
            next_node = f"J{i+1}"
            main_pipe_edges.append((current_node, next_node))
            current_node = next_node
        main_pipe_edges.append((current_node, tank_id))
        
        # Find an edge with elevation difference for optimal pump placement
        pump_placed = False
        for edge in main_pipe_edges:
            node1, node2 = edge
            if node1 in self.G.nodes and node2 in self.G.nodes:
                pos1 = self.node_positions[node1]
                pos2 = self.node_positions[node2]
                elev1 = self.get_elevation(pos1[0], pos1[1])
                elev2 = self.get_elevation(pos2[0], pos2[1])
                
                # If we have a significant elevation difference, place pump
                if abs(elev1 - elev2) > 5:
                    pump_id = "P1"
                    self.pumps.append(pump_id)
                    # Remove direct connection
                    self.G.remove_edge(node1, node2)
                    # Add pump and connect
                    self.G.add_node(pump_id, type='pump', power=20)
                    self.G.add_edge(node1, pump_id)
                    self.G.add_edge(pump_id, node2)
                    
                    # Place pump between the junctions
                    pos1 = self.node_positions[node1]
                    pos2 = self.node_positions[node2]
                    self.node_positions[pump_id] = ((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2)
                    
                    self.pipe_diameters[(node1, pump_id)] = 300  # Keep the large diameter for main pipe
                    self.pipe_diameters[(pump_id, node2)] = 300  # Keep the large diameter for main pipe
                    pump_placed = True
                    break
        
        # If no suitable elevation difference was found, place pump in the middle of main pipe
        if not pump_placed:
            # Choose a central edge on the main pipe
            central_edge_index = len(main_pipe_edges) // 2
            if central_edge_index < len(main_pipe_edges):
                node1, node2 = main_pipe_edges[central_edge_index]
                pump_id = "P1"
                self.pumps.append(pump_id)
                # Remove direct connection
                self.G.remove_edge(node1, node2)
                # Add pump and connect
                self.G.add_node(pump_id, type='pump', power=15)
                self.G.add_edge(node1, pump_id)
                self.G.add_edge(pump_id, node2)
                
                # Place pump between the junctions
                pos1 = self.node_positions[node1]
                pos2 = self.node_positions[node2]
                self.node_positions[pump_id] = ((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2)
                
                self.pipe_diameters[(node1, pump_id)] = 300  # Keep the large diameter for main pipe
                self.pipe_diameters[(pump_id, node2)] = 300  # Keep the large diameter for main pipe

        return self.G, self.node_positions

    def find_nearest_node(self, point, node_type=None):
        """Find the nearest node to a given point, optionally filtering by node type"""
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.G.nodes():
            if node_type and self.G.nodes[node].get('type') != node_type:
                continue
                
            if node in self.node_positions:
                pos = self.node_positions[node]
                dist = ((pos[0] - point[0])**2 + (pos[1] - point[1])**2)**0.5
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
                    
        return nearest_node
    
    def generate_network(self):
        """Generate a network based on the specified type"""
        # Generate elevation map first
        self.generate_elevation_map()
        
        if self.network_type == 'looped':
            return self.generate_looped_network()
        else:  # branched
            return self.generate_branched_network()
    
    def plot_network(self, show_elevation=True, show_3d=False, results=None):
        """
        Plot the network with optional elevation map and hydraulic results visualization
        
        Args:
            show_elevation (bool): Whether to show the elevation map
            show_3d (bool): Whether to show a 3D visualization
            results (dict): Optional dictionary containing:
                - 'pressure_deficit': Dict mapping node IDs to pressure deficit values
                - 'headloss': Dict mapping pipe tuples (node1, node2) to headloss values
        """
        if not self.G:
            print("No network to plot. Generate a network first.")
            return
        
        if show_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot elevation map as surface
            if show_elevation and self.elevation_map:
                X = self.elevation_map['X']
                Y = self.elevation_map['Y']
                Z = self.elevation_map['Z']
                
                # Downsample for performance
                stride = 5
                surf = ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
                                    cmap=cm.terrain, alpha=0.6, linewidth=0, antialiased=True)
            
            # Plot edges (pipes)
            headloss_values = []
            if results and 'headloss' in results:
                for edge in self.G.edges():
                    if edge in results['headloss'] or (edge[1], edge[0]) in results['headloss']:
                        headloss = results['headloss'].get(edge, results['headloss'].get((edge[1], edge[0]), 0))
                        headloss_values.append(headloss)
                
                if headloss_values:
                    vmin = min(headloss_values)
                    vmax = max(headloss_values)
                    headloss_norm = plt.Normalize(vmin, vmax)
                    headloss_cmap = plt.cm.ScalarMappable(norm=headloss_norm, cmap='plasma')
            
            for edge in self.G.edges():
                node1, node2 = edge
                if node1 in self.node_positions and node2 in self.node_positions:
                    x1, y1 = self.node_positions[node1]
                    x2, y2 = self.node_positions[node2]
                    
                    z1 = 0
                    z2 = 0
                    
                    if 'elevation' in self.G.nodes[node1]:
                        z1 = self.G.nodes[node1]['elevation']
                    elif 'base_head' in self.G.nodes[node1]:
                        z1 = self.G.nodes[node1]['base_head']
                    
                    if 'elevation' in self.G.nodes[node2]:
                        z2 = self.G.nodes[node2]['elevation']
                    elif 'base_head' in self.G.nodes[node2]:
                        z2 = self.G.nodes[node2]['base_head']
                    
                    # Get pipe diameter for line width
                    diameter = self.pipe_diameters.get((node1, node2), 100)
                    line_width = diameter / 50  # Scale for visualization
                    
                    # Apply color based on headloss if available
                    if results and 'headloss' in results:
                        if edge in results['headloss'] or (edge[1], edge[0]) in results['headloss']:
                            headloss = results['headloss'].get(edge, results['headloss'].get((edge[1], edge[0]), 0))
                            pipe_color = headloss_cmap.to_rgba(headloss)
                        else:
                            pipe_color = 'black'
                    else:
                        pipe_color = 'black'
                    
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color=pipe_color, linewidth=line_width)
            
            # Add headloss colorbar if needed
            if results and 'headloss' in results and headloss_values:
                cbar = fig.colorbar(headloss_cmap, ax=ax, label='Headloss (m)')
            
            # Plot nodes with pressure deficit information if available
            pressure_values = []
            if results and 'pressure_deficit' in results:
                for node, deficit in results['pressure_deficit'].items():
                    if node in self.G.nodes and self.G.nodes[node].get('type') == 'junction':
                        pressure_values.append(deficit)
                        
                if pressure_values:
                    vmin = min(pressure_values)
                    vmax = max(pressure_values)
                    pressure_norm = plt.Normalize(vmin, vmax)
                    pressure_cmap = plt.cm.RdYlGn_r  # Red for high deficit, green for low deficit
            
            for node in self.G.nodes():
                if node in self.node_positions:
                    x, y = self.node_positions[node]
                    node_type = self.G.nodes[node].get('type', 'unknown')
                    
                    if node_type == 'junction':
                        z = self.G.nodes[node].get('elevation', 0)
                        
                        # Set node color based on pressure deficit if available
                        if results and 'pressure_deficit' in results and node in results['pressure_deficit']:
                            deficit = results['pressure_deficit'][node]
                            node_color = pressure_cmap(pressure_norm(deficit))
                            label_text = f"{node}: {deficit:.1f}m"
                        else:
                            node_color = 'blue'
                            label_text = node
                        
                        ax.scatter([x], [y], [z], c=[node_color], s=100, marker='o', 
                                label='Junction' if node == self.junctions[0] else "")
                        
                    elif node_type == 'reservoir':
                        z = self.G.nodes[node].get('base_head', 0)
                        ax.scatter([x], [y], [z], c='green', s=200, marker='s', label='Reservoir')
                    elif node_type == 'tank':
                        z = self.G.nodes[node].get('elevation', 0) + self.G.nodes[node].get('initial_level', 0)
                        ax.scatter([x], [y], [z], c='purple', s=200, marker='^', label='Tank')
                    elif node_type == 'pump':
                        # Place pump at average elevation of connected nodes
                        neighbors = list(self.G.neighbors(node))
                        z = 0
                        count = 0
                        for neighbor in neighbors:
                            if 'elevation' in self.G.nodes[neighbor]:
                                z += self.G.nodes[neighbor]['elevation']
                                count += 1
                        if count > 0:
                            z /= count
                        ax.scatter([x], [y], [z], c='red', s=150, marker='*', label='Pump')
            
            # Add pressure deficit colorbar if needed
            if results and 'pressure_deficit' in results and pressure_values:
                pressure_sm = plt.cm.ScalarMappable(norm=pressure_norm, cmap=plt.cm.RdYlGn_r)
                cbar = fig.colorbar(pressure_sm, ax=ax, label='Pressure Deficit (m)')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Elevation (m)')
            ax.set_title(f'{self.network_type.capitalize()} Hydraulic Network')
            
            # Only show legend once for each type
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        else:  # 2D plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot elevation map as contour
            if show_elevation and self.elevation_map:
                X = self.elevation_map['X']
                Y = self.elevation_map['Y']
                Z = self.elevation_map['Z']
                
                contour = ax.contourf(X, Y, Z, cmap=cm.terrain, alpha=0.5, levels=20)
                fig.colorbar(contour, ax=ax, label='Elevation (m)')
            
            # Prepare colormaps for headloss
            headloss_values = []
            if results and 'headloss' in results:
                for edge in self.G.edges():
                    if edge in results['headloss'] or (edge[1], edge[0]) in results['headloss']:
                        headloss = results['headloss'].get(edge, results['headloss'].get((edge[1], edge[0]), 0))
                        headloss_values.append(headloss)
                
                if headloss_values:
                    vmin = min(headloss_values)
                    vmax = max(headloss_values)
                    headloss_norm = plt.Normalize(vmin, vmax)
                    headloss_cmap = plt.cm.ScalarMappable(norm=headloss_norm, cmap='plasma')
                    cbar = fig.colorbar(headloss_cmap, ax=ax, label='Headloss (m)')
            
            # Plot edges (pipes)
            for edge in self.G.edges():
                node1, node2 = edge
                if node1 in self.node_positions and node2 in self.node_positions:
                    x1, y1 = self.node_positions[node1]
                    x2, y2 = self.node_positions[node2]
                    
                    # Get pipe diameter for line width
                    diameter = self.pipe_diameters.get((node1, node2), 100)
                    line_width = diameter / 50  # Scale for visualization
                    
                    # Apply color based on headloss if available
                    if results and 'headloss' in results:
                        if edge in results['headloss'] or (edge[1], edge[0]) in results['headloss']:
                            headloss = results['headloss'].get(edge, results['headloss'].get((edge[1], edge[0]), 0))
                            pipe_color = headloss_cmap.to_rgba(headloss)
                        else:
                            pipe_color = 'black'
                    else:
                        pipe_color = 'black'
                    
                    ax.plot([x1, x2], [y1, y2], color=pipe_color, linewidth=line_width)
            
            # Prepare colormaps for pressure deficit
            pressure_values = []
            if results and 'pressure_deficit' in results:
                for node, deficit in results['pressure_deficit'].items():
                    if node in self.G.nodes and self.G.nodes[node].get('type') == 'junction':
                        pressure_values.append(deficit)
                
                if pressure_values:
                    vmin = min(pressure_values)
                    vmax = max(pressure_values)
                    pressure_norm = plt.Normalize(vmin, vmax)
                    pressure_cmap = plt.cm.RdYlGn_r  # Red for high deficit, green for low deficit
                    pressure_sm = plt.cm.ScalarMappable(norm=pressure_norm, cmap=pressure_cmap)
                    cbar = fig.colorbar(pressure_sm, ax=ax, label='Pressure Deficit (m)')
            
            # Plot nodes
            for node in self.G.nodes():
                if node in self.node_positions:
                    x, y = self.node_positions[node]
                    node_type = self.G.nodes[node].get('type', 'unknown')
                    
                    if node_type == 'junction':
                        # Set node color based on pressure deficit if available
                        if results and 'pressure_deficit' in results and node in results['pressure_deficit']:
                            deficit = results['pressure_deficit'][node]
                            node_color = pressure_cmap(pressure_norm(deficit))
                            label_text = f"{node}: {deficit:.1f}m"
                        else:
                            node_color = 'blue'
                            label_text = node
                        
                        ax.scatter(x, y, c=[node_color], s=100, marker='o', 
                                label='Junction' if node == self.junctions[0] else "", zorder=5)
                        ax.text(x, y-20, label_text, fontsize=8, ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7))
                    elif node_type == 'reservoir':
                        ax.scatter(x, y, c='green', s=200, marker='s', label='Reservoir', zorder=5)
                        ax.text(x, y-20, node, fontsize=10, fontweight='bold', ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7))
                    elif node_type == 'tank':
                        ax.scatter(x, y, c='purple', s=200, marker='^', label='Tank', zorder=5)
                        ax.text(x, y-20, node, fontsize=10, fontweight='bold', ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7))
                    elif node_type == 'pump':
                        ax.scatter(x, y, c='red', s=150, marker='*', label='Pump', zorder=5)
                        ax.text(x, y-20, node, fontsize=10, fontweight='bold', ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{self.network_type.capitalize()} Hydraulic Network')
            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(0, self.grid_size)
            
            # Only show legend once for each type
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        plt.tight_layout()
        return fig

    def get_network_data(self):
        """Return a dictionary with all network data in a structured format"""
        data = {
            'network_type': self.network_type,
            'nodes': {},
            'links': []
        }
        
        # Get node data
        for node in self.G.nodes():
            node_data = self.G.nodes[node].copy()
            node_data['position'] = self.node_positions.get(node, (0, 0))
            data['nodes'][node] = node_data
        
        # Get link (pipe) data
        for edge in self.G.edges():
            node1, node2 = edge
            link_data = {
                'start_node': node1,
                'end_node': node2,
                'diameter': self.pipe_diameters.get((node1, node2), 100)
            }
            data['links'].append(link_data)
            
        return data

    def export_to_inp(self, filename):
        """Export the network to EPANET .inp format"""
        # This would need to be implemented based on EPANET format
        # This is a placeholder for future implementation
        pass


def generate_hydraulic_network(network_type='looped', num_junctions=10, grid_size=1000, seed=None):
    """
    Generate a hydraulic network for simulation
    
    Args:
        network_type (str): 'looped' or 'branched'
        num_junctions (int): Number of junctions in the network
        grid_size (int): Size of the grid (meters)
        seed (int): Random seed for reproducibility
        
    Returns:
        HydraulicNetwork: The generated network object
    """
    network = HydraulicNetwork(network_type, num_junctions, grid_size, seed)
    network.generate_network()
    return network


# Example usage
if __name__ == "__main__":

    # If no folder 'network visualisation' exists, create it
    if not os.path.exists('./PPO_Optimisation_2/Network_visualisation'):
        os.makedirs('./PPO_Optimisation_2/Network_visualisation')

    # Generate a looped network
    looped_network = generate_hydraulic_network(network_type='looped', num_junctions=25, seed=2)
    fig1 = looped_network.plot_network(show_elevation=True)
    
    # Generate a branched network
    branched_network = generate_hydraulic_network(network_type='branched', num_junctions=25, seed=2)
    fig2 = branched_network.plot_network(show_elevation=True)
    
    # Show 3D visualization
    fig3 = looped_network.plot_network(show_elevation=True, show_3d=True)
    fig4 = branched_network.plot_network(show_elevation=True, show_3d=True)

    # Save the figures to network visualisation folder
    fig1.savefig('./PPO_Optimisation_2/Network_visualisation/looped_network.png')
    fig2.savefig('./PPO_Optimisation_2/Network_visualisation/branched_network.png')
    fig3.savefig('./PPO_Optimisation_2/Network_visualisation/looped_network_3D.png')
    fig4.savefig('./PPO_Optimisation_2/Network_visualisation/branched_network_3D.png')
    
    plt.show()