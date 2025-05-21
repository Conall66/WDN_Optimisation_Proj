
# Takes wntr model and converts to networkx to visualise

# Import necessary libraries
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import wntr
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def visualise_wntr(wn, elevation_map, results=None, title="Water Distribution Network"):
    """
    Visualize the water network with pressure or flow information if available.
    
    Parameters:
    -----------
    wn : WNTR model
        The water network topology
    results : dict, optional
        Dictionary containing 'pressure' and 'headloss' data
        'pressure' should be a dict mapping node IDs to pressure values
        'headloss' should be a dict mapping edge tuples (u,v) to head loss values
    title : str
        Plot title
    """
    
    G = wn.get_graph()
    # Get node coordinates
    pos = wn.query_node_attribute('coordinates')

    # Get pressure results for the first timestep (index 0)
    pressure = results.node['pressure'].iloc[0].to_dict()
    
    # Get headloss results for the first timestep
    headloss = results.link['headloss'].iloc[0].to_dict()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Add elevationmapas background
    if elevation_map is not None:
        ax.imshow(elevation_map, cmap='terrain', extent=(0, elevation_map.shape[1], 0, elevation_map.shape[0]), alpha=0.5)
        ax.set_title(title + " - Elevation Map", fontsize=16)
    
    # Create node lists by type
    junction_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Junction']
    reservoir_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Reservoir']
    tank_nodes = [node_name for node_name, node in wn.nodes() if node.node_type == 'Tank']
    
    # Get edges (pipes and pumps)
    pipes = [link_name for link_name, link in wn.links() if link.link_type == 'Pipe']
    pumps = [link_name for link_name, link in wn.links() if link.link_type == 'Pump']

    # max_diameter = max(wn.get_link_attribute('diameter').values())
    # min_diameter = min(wn.get_link_attribute('diameter').values())
    # max_edge_width = 5
    # min_edge_width = 1
    # scale_factor = (max_edge_width - min_edge_width) / (max_diameter - min_diameter)    
    
    # Draw pipes with width proportional to diameter
    for pipe in pipes:
        link = wn.get_link(pipe)
        start_node = link.start_node_name
        end_node = link.end_node_name
        diameter = link.diameter
        
        # Scale diameter to a reasonable width (adjust scaling factor as needed)
        width = diameter * 0.01  # You can adjust this multiplier based on your preference
        
        # Draw the pipe with scaled width
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(start_node, end_node)],
                              width=width, edge_color='gray', style='solid')
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(wn.get_link(p).start_node_name, wn.get_link(p).end_node_name) for p in pumps],
                          width=width, edge_color='red', style='dashed')
    
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