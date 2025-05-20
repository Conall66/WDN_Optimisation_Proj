import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def visualise_network(graph, elevation_map = None, results=None, title="Water Distribution Network"):
    """
    Visualize the water network with pressure or flow information if available.
    
    Parameters:
    -----------
    G : NetworkX graph
        The water network topology
    results : dict, optional
        Dictionary containing 'pressure' and 'headloss' data
        'pressure' should be a dict mapping node IDs to pressure values
        'headloss' should be a dict mapping edge tuples (u,v) to head loss values
    title : str
        Plot title
    """

    # plt.figure(figsize=(12, 10))
    plt.figure()
    plt.title(title)

    # Get the coordinate range from the graph nodes
    pos = nx.get_node_attributes(graph, 'coordinates')
    x_coords = [xy[0] for xy in pos.values()]
    y_coords = [xy[1] for xy in pos.values()]

    if elevation_map is not None:

        min_x, max_x = 0, elevation_map.shape[0]
        min_y, max_y = 0, elevation_map.shape[1]
        
        # Display the elevation map with the correct extent
        # Important: match the extent to the area_size used in network generation
        plt.imshow(elevation_map, origin='lower', cmap='terrain', alpha=0.5, 
                extent=[min_x, max_x, min_y, max_y])
        plt.colorbar(label='Elevation (m)', shrink=0.7)
    
    # Display the elevation map with the correct extent
    # plt.imshow(elevation_map, origin='lower', cmap='terrain', alpha=0.5, extent=[0, elevation_map.shape[0], 0, elevation_map.shape[1]])
    # plt.colorbar(label='Elevation (m)', shrink=0.7)

    if results is None:
        # Original visualization without hydraulic data
        # Reservoir nodes are blue circles, tanks are green squares, junctions are red triangles
        node_colors = []
        node_shapes = []
        for node, data in graph.nodes(data=True):
            if data['type'] == 'reservoir':
                node_colors.append('blue')
                node_shapes.append('^')
            elif data['type'] == 'tank':
                node_colors.append('green')
                node_shapes.append('s')
            elif data['type'] == 'commercial':
                node_colors.append('red')
                node_shapes.append('o')
            elif data['type'] == 'residential':
                node_colors.append('purple')
                node_shapes.append('o')
            else:
                node_colors.append('gray')
                node_shapes.append('o')
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_shape='o')

        max_width = max(graph.edges(data=True), key=lambda x: x[2]['diameter'])[2]['diameter']
        min_width = min(graph.edges(data=True), key=lambda x: x[2]['diameter'])[2]['diameter']
        # Scale widths so largest value has a width of 3 and smallest has a width of 1
        scale_factor = (max_width - min_width)/2
        # Draw edges with varying widths based on diameter
        for u, v, data in graph.edges(data=True):
            if 'pump_type' not in data:
                width = 1 + (data['diameter'] - min_width) / scale_factor
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=width, edge_color='gray')
            else:
                # Draw pump edges in a different color
                width = 1 + (data['diameter'] - min_width) / scale_factor
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=width, edge_color='red')
        
        # Draw normal pipes (edges without pump attributes)
        # normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' not in d]
        # nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, edge_color='gray')
        
        # # Draw pump edges in a different color
        # pump_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' in d]
        # nx.draw_networkx_edges(graph, pos, edgelist=pump_edges, edge_color='red', width=2.0)
        
        # Add legend
        legend_labels = {
            'Reservoir': 'blue',
            'Tank': 'green',
            'Commerical Junction': 'red',
            'Residential Junction': 'purple'
        }
        for label, color in legend_labels.items():
            marker = '_' if label == 'Pump' else 'o'
            plt.scatter([], [], color=color, label=label, s=100)
        
    else:
        # Visualization with hydraulic data
        # 1. Node coloring based on pressure
        # if 'pressure' in results:
        pressures = []
        junction_nodes = []

        pressure_data = results.node['pressure']
        
        for node, data in graph.nodes(data=True):
            if node in pressure_data.columns and data['type'] == 'residential' or data['type'] == 'commercial':
                junction_nodes.append(node)
                pressures.append(np.mean(pressure_data[node]))
        
        if pressures:  # Only if we have pressure values
            # Create a colormap for pressures
            node_cmap = plt.cm.viridis
            norm_pressure = Normalize(vmin=min(pressures), vmax=max(pressures))
            
            # Draw junction nodes with pressure coloring
            junction_colors = [node_cmap(norm_pressure(np.mean(pressure_data[node]))) for node in junction_nodes]
            nx.draw_networkx_nodes(graph, pos, nodelist=junction_nodes, node_color=junction_colors, node_shape='o')
            
            # Add colorbar for pressure
            sm_pressure = ScalarMappable(cmap=node_cmap, norm=norm_pressure)
            sm_pressure.set_array([])
            cbar_pressure = plt.colorbar(sm_pressure, ax=plt.gca(), shrink=0.6, pad=0.01)
            cbar_pressure.set_label('Pressure')
        
        # Draw other nodes with standard colors
        reservoirs = [node for node, data in graph.nodes(data=True) if data['type'] == 'reservoir']
        tanks = [node for node, data in graph.nodes(data=True) if data['type'] == 'tank']
        
        nx.draw_networkx_nodes(graph, pos, nodelist=reservoirs, node_color='blue', node_shape='o')
        nx.draw_networkx_nodes(graph, pos, nodelist=tanks, node_color='green', node_shape='s')
    
        # 2. Edge coloring based on head loss
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' not in d]
        pump_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' in d]
        
        # 2. Edge coloring based on head loss
        headloss_data = results.link['headloss']
        pipe_edges = []
        pipe_headlosses = {}
        
        # Map graph edges to WNTR pipe IDs (this is crucial for correct mapping)
        edge_to_pipe_id = {}
        pipe_count = 1
        for u, v, data in graph.edges(data=True):
            edge_to_pipe_id[(u, v)] = str(pipe_count)
            edge_to_pipe_id[(v, u)] = str(pipe_count)  # Account for undirected graph
            pipe_count += 1
        
        # Collect headloss values for all pipes
        for u, v in graph.edges():
            pipe_id = edge_to_pipe_id.get((u, v))
            if pipe_id and pipe_id in headloss_data:
                pipe_edges.append((u, v))
                # For static simulation, get the first time step
                pipe_headlosses[(u, v)] = abs(headloss_data[pipe_id][0])
        
        if pipe_edges:  # Only if we have pipe data
            # Create a colormap for head loss
            edge_cmap = plt.cm.cool
            headloss_values = list(pipe_headlosses.values())
            norm_headloss = Normalize(vmin=min(headloss_values), vmax=max(headloss_values))

        # Draw edges with head loss coloring
            for u, v in pipe_edges:
                nx.draw_networkx_edges(graph, pos, 
                                     edgelist=[(u, v)], 
                                     edge_color=[edge_cmap(norm_headloss(pipe_headlosses[(u, v)]))], 
                                     width=2.0)
            
            # Add colorbar for head loss
            sm_headloss = ScalarMappable(cmap=edge_cmap, norm=norm_headloss)
            sm_headloss.set_array([])
            cbar_headloss = plt.colorbar(sm_headloss, ax=plt.gca(), shrink=0.6, pad=0.05)
            cbar_headloss.set_label('Head Loss (m)')
            
    
        # Legend for hydraulic visualization
        plt.scatter([], [], color='blue', label='Reservoir', s=100)
        plt.scatter([], [], color='green', label='Tank', s=100)
        plt.scatter([], [], marker='o', color='gray', label='Junction', s=100)
        # plt.plot([], [], color='red', label='Pump', linewidth=2.5)

    # Legend for hydraulic visualization
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Reservoir'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Tank'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Junction'),
            plt.Line2D([0], [0], color='red', lw=2, label='Pump')
        ]
        plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()