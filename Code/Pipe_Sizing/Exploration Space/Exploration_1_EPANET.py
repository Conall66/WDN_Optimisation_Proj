
from epyt import epanet
import numpy as np
import matplotlib.pyplot as plt
import random

d = epanet('Testfile.inp', 'CREATE')
    
# Corrected node definitions with unique keys
nodes = {
    'Node 1': {
        'NodeID': 'Node1',
        'x': 0,
        'y': 0,
        'type': 'Source',
        'demand': 0,
        'head': 50,
        'elevation': 10,
    },
    'Node 2': {
        'NodeID': 'Node2',
        'x': 0,
        'y': 2,
        'type': 'Junction',
        'demand': 0,
        'elevation': 10,
    },
    'Node 3': {
        'NodeID': 'Node3',
        'x': 2,
        'y': 2,
        'type': 'Junction',
        'demand': 0,
        'elevation': 10,
    },
    'Node 4': {
        'NodeID': 'Node4',
        'x': 3,
        'y': 0,
        'type': 'Junction',
        'demand': 0,
        'elevation': 10,
    },
    'Node 5': {
        'NodeID': 'Node5',
        'x': 3,
        'y': 2,
        'type': 'Tank',
        'MinLevel': 0,
        'MaxLevel': 10,
        'InitialLevel': 5,
        'Diameter': 50,
        'elevation': 10,
        'Volume': 0,
    },
}

# Add reservoir

success = d.addNodeJunction('Junct 1', [0, 0])
if not success:
    print("Node was not added")

d.addNodeReservoir(nodes['Node 1']['NodeID'], [nodes['Node 1']['x'], nodes['Node 1']['y']])

# Add junctions
for node_key in ['Node 2', 'Node 3', 'Node 4']:
    node = nodes[node_key]
    d.addNodeJunction(node['NodeID'], [node['x'], node['y']], node['elevation'], node['demand'])

# Add tank - use the direct tank constructor approach
# Parameters: ID, elevation, initial level, min level, max level, diameter, min volume
d.addNodeTank('Node5', 
                [nodes['Node 5']['x'], nodes['Node 5']['y']],
                nodes['Node 5']['elevation'],
            #  nodes['Node 5']['InitialLevel'],
            #  nodes['Node 5']['MinLevel'], 
            #  nodes['Node 5']['MaxLevel'],
            #  nodes['Node 5']['Diameter'],
            #  nodes['Node 5']['Volume']
            )

# Arbitrarily connect existing network
d.addLinkPipe('Pipe1', 'Node1', 'Node2', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe2', 'Node2', 'Node3', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe3', 'Node3', 'Node4', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe4', 'Node2', 'Node5', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe5', 'Node3', 'Node5', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe6', 'Node4', 'Node5', 100, 10, 0.1, 0)
d.addLinkPipe('Pipe7', 'Node1', 'Node5', 100, 10, 0.1, 0)

# display information about d
print(f"Node Count: {d.getNodeCount()}")
d.plot()
d.plot_show()

# Save to file
d.saveInputFile('Testfile.inp')