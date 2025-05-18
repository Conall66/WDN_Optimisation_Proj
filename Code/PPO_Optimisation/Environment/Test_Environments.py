
"""

In this file, we generate basic looped and branched networks for testing hydraulic modelling. These networks are hardcoded to ensure a degree of viabiity.

"""

def create_looped_network(size, area):
    """
    Create a simple looped water distribution network.

    Parameters:
    -----------
    size : int
        The number of junctions in the network.
    area : int 
        The area of the network.
    
    Returns:
        graph : networkx.Graph
            The generated looped water distribution network.
    """
    import networkx as nx

    # Create a directed graph
    graph = nx.graph()

    # Add reservoir in the top left
    graph.add_node(1, type='Reservoir', position=(0, area[1]))

    # Add tank in the top right corner
    graph.add_node(2, type='Tank', position=(area[0], area[1]))

    # Add junctions in a loop
    if size = 'small':
        num_nodes = 10
    elif size = 'medium':
        num_nodes = 20
    elif size = 'large':
        num_nodes = 50



    return graph