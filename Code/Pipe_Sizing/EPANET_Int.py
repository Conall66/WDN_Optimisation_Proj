
"""

In this file, we take a network of pipes and nodes and create a .inp file for EPANET. We then run EPANET and read the results back into Python. The .inp file is created using the EPANET toolkit, which is a C library that allows for the simulation of water distribution systems. The results are read back into Python using the EPANET toolkit as well.

"""

# Import libraries to convert the networkx graph to a .inp file for EPANET
import os
import sys
import numpy as np

# Generate .inp file

pipe_diameters = [0.1, 0.2, 0.3, 0.4, 0.5]  # Arbitrary pipe diameters in meters
pipe_roughnesses = [0.01, 0.02, 0.03, 0.04, 0.05]  # Arbitrary pipe roughnesses in meters

def generate_inp_file(G, filename, pipe_diameters, pipe_roughnesses):
    """
    Generate an EPANET .inp file from a networkx graph.
    """
    with open(filename, 'w') as f:
        # Write the header
        f.write("[TITLE]\n")
        f.write("Generated EPANET input file\n")
        f.write("\n")
        
        # Write the nodes
        f.write("[JUNCTIONS]\n")
        for node in G.nodes(data=True):
            if node[1]['source']:
                f.write(f"{node[0][0]} {node[0][1]} 0.0 0.0 0.0 0.0\n")
            else:
                f.write(f"{node[0][0]} {node[0][1]} {node[1]['demand']} 0.0 0.0 0.0\n")
        f.write("\n")
        
        # Write the pipes
        f.write("[PIPES]\n")
        for u, v, data in G.edges(data=True):
            diameter = np.random.choice(pipe_diameters)
            roughness = np.random.choice(pipe_roughnesses)
            f.write(f"{u[0]} {v[0]} {diameter} {roughness} 100.0 100.0\n")
        f.write("\n")
        
        # Write the end of the file
        f.write("[END]\n")

