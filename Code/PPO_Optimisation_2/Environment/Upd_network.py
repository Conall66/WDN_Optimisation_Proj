
"""

In this file, we take as input the existing graph and some conditions on the nature of its evolution, including demand forecasting, expansion type and time step. We then generate a new graph starting from the same topology, but with the potential for additional nodes to be added to the system. 

"""

# Import neccesary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import wntr

from Visualise_network import visualise_network
from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance

# For each file in the networks folder, create a subfolder with the same name

# Tets import a single file and visualise
# file_1 = os.path.join('Newtorks', 'exeter', 'hanoi-3.inp')
# Check file exist in directory

script = os.path.dirname(__file__)
file_1 = os.path.join(script, 'Networks', 'exeter', 'hanoi-3.inp')
if not os.path.exists(file_1):
    raise FileNotFoundError(f"The file {file_1} does not exist.")

# Load the network
wn = wntr.network.WaterNetworkModel(file_1)
# Display the network

# figure = plt.figure(figsize=(10, 10))
wntr.graphics.plot_network(wn, title="Water Distribution Network hanoi-3")
plt.show()

# Get hydraulic results from the network
results = run_epanet_simulation(wn)

print(f"Average pressure: {results.node['pressure'].mean()}")
print(f"Minimum pressure: {results.node['pressure'].min()}")
print(f"Maximum pressure: {results.node['pressure'].max()}")
print("-------------------------------------------")

print(f"Average headloss: {results.link['headloss'].mean()}")
print(f"Minimum headloss: {results.link['headloss'].min()}")
print(f"Maximum headloss: {results.link['headloss'].max()}")
print("-------------------------------------------")

# Get hydraulic performance
metrics = evaluate_network_performance(wn, results)

# Save the results to a CSV file
# results_df = pd.DataFrame(results.node['pressure'].values, columns=wn.node_name_list)
# results_df.to_csv(os.path.join(script, 'Network_Performance_Metrics', 'Initial_inp_pfm.csv'), index=False)

# Visualise the network
save_dir = os.path.join(script, 'Network_visualisation')
save_path = os.path.join(script, 'Network_visualisation', 'anystown-3.png')

# visualise_network(wn, results, title="Water Distribution Network hanoi-3", save_path=save_path, mode = '3d')

# Basic topology visualization
# plt.figure(figsize=(12, 10))

# Pressure visualization
# plt.figure(figsize=(12, 10))
pressure = results.node['pressure'].iloc[0]
wntr.graphics.plot_network(wn, node_attribute=pressure, title='Node Pressure (m)', 
                          node_colorbar_label='Pressure (m)', node_size = 100)
plt.savefig(os.path.join(save_dir, 'pressure.png'), dpi=300)
plt.show()
# Headloss visualisation
# plt.figure(figsize=(12, 10))
headloss = results.link['headloss'].iloc[0]
wntr.graphics.plot_network(wn, link_attribute=headloss, title='Pipe Headloss (m)',
                          link_colorbar_label='Headloss (m)', node_size = 100)
plt.savefig(os.path.join(save_dir, 'headloss.png'), dpi=300)
plt.show()

# Rename the file the same name + 'start'

# Function to update the network given input parameters: