
"""

In this file, we choose networks from the imported networks to add so Networks subfolder

"""

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
small_networks_path = os.path.join(script_dir, "Imported_networks", "epanet-example-networks", "epanet-tests", "small")
destination_path = os.path.join(script_dir, "Networks")

input_path = os.path.join(script_dir, "Imported_networks", "epanet-tests", "small")
destination_path = os.path.join(script_dir, "Networks")

if os.path.exists(small_networks_path):
    print(f"Small networks path exists: {small_networks_path}")
else:
    print(f"Small networks path does not exist: {small_networks_path}")

for file in os.listdir(small_networks_path):
    if file.endswith(".inp"):
        source_file = os.path.join(small_networks_path, file)
        destination_file = os.path.join(destination_path, file)
        shutil.copy(source_file, destination_file)
        print(f"Copied {file} to {destination_path}")


