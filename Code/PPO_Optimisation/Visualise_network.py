import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def visualise_network(graph, results=None, title="Water Distribution Network"):
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

    plt.figure(figsize=(10, 10))
    plt.title(title)
    pos = nx.get_node_attributes(graph, 'position')

    if results is None:
        # Original visualization without hydraulic data
        # Reservoir nodes are blue circles, tanks are green squares, junctions are red triangles
        node_colors = []
        node_shapes = []
        for node, data in graph.nodes(data=True):
            if data['type'] == 'Reservoir':
                node_colors.append('blue')
                node_shapes.append('o')
            elif data['type'] == 'Tank':
                node_colors.append('green')
                node_shapes.append('s')
            elif data['type'] == 'Junction':
                node_colors.append('red')
                node_shapes.append('o')
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_shape='o')
        
        # Draw normal pipes (edges without pump attributes)
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' not in d]
        nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, edge_color='gray')
        
        # Draw pump edges in a different color
        pump_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' in d]
        nx.draw_networkx_edges(graph, pos, edgelist=pump_edges, edge_color='red', width=2.0)
        
        # Add legend
        legend_labels = {
            'Reservoir': 'blue',
            'Tank': 'green',
            'Junction': 'red',
            'Pump': 'purple'
        }
        for label, color in legend_labels.items():
            plt.scatter([], [], color=color, label=label, s=100)
        
    else:
        # Visualization with hydraulic data
        # 1. Node coloring based on pressure
        # if 'pressure' in results:
        pressures = []
        junction_nodes = []

        pressure_data = results.node['pressure']
        
        for node, data in graph.nodes(data=True):
            if node in pressure_data.columns and data['type'] == 'Junction':
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
        reservoirs = [node for node, data in graph.nodes(data=True) if data['type'] == 'Reservoir']
        tanks = [node for node, data in graph.nodes(data=True) if data['type'] == 'Tank']
        
        nx.draw_networkx_nodes(graph, pos, nodelist=reservoirs, node_color='blue', node_shape='o')
        nx.draw_networkx_nodes(graph, pos, nodelist=tanks, node_color='green', node_shape='s')
    
        # 2. Edge coloring based on head loss
        normal_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' not in d]
        pump_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'pump_type' in d]
        
        # Collect head loss values for normal pipes
        edge_headlosses = []
        pipe_edges = []

        headloss_data = results.link['headloss']
        
        for u, v in normal_edges:
            link_id = graph[u][v].get('edge_id')
            if link_id in headloss_data.columns:
                pipe_edges.append((u, v))
                edge_headlosses.append(abs(np.mean(headloss_data[link_id])))
        
        if edge_headlosses:  # Only if we have headloss values
            # Create a colormap for head loss
            edge_cmap = plt.cm.cool
            min_headloss = min(edge_headlosses) if edge_headlosses else 0
            max_headloss = max(edge_headlosses) if edge_headlosses else 1
            norm_headloss = Normalize(vmin=min_headloss, vmax=max_headloss)
            
            # Draw edges with head loss coloring
            for i, (u, v) in enumerate(pipe_edges):
                edge_color = edge_cmap(norm_headloss(edge_headlosses[i]))
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color=[edge_color], width=2.0)
            
            # Add colorbar for head loss
            sm_headloss = ScalarMappable(cmap=edge_cmap, norm=norm_headloss)
            sm_headloss.set_array([])
            cbar_headloss = plt.colorbar(sm_headloss, ax=plt.gca(), shrink=0.6, pad=0.05)
            cbar_headloss.set_label('Head Loss')
    
        # Legend for hydraulic visualization
        plt.scatter([], [], color='blue', label='Reservoir', s=100)
        plt.scatter([], [], color='green', label='Tank', s=100)
        plt.scatter([], [], marker='o', color='gray', label='Junction', s=100)
        # plt.plot([], [], color='red', label='Pump', linewidth=2.5)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(graph, 'edge_id')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_weight='bold')
    
    plt.legend()
    plt.show()