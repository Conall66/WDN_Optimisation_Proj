
"""

Test that new networks have the correct number of pipes in each stepo for each scenario

"""

import os
import matplotlib.pyplot as plt
from wntr.network import WaterNetworkModel

# --- Configuration ---
networks_folder = 'Modified_nets'
scenarios = [s for s in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, s))]

# --- Iterate through each scenario ---
for scenario in scenarios:
    scenario_path = os.path.join(networks_folder, scenario)
    inp_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.inp')])

    num_junctions = []
    num_pipes = []
    time_steps = []

    for idx, inp_file in enumerate(inp_files):
        full_path = os.path.join(scenario_path, inp_file)
        try:
            wn = WaterNetworkModel(full_path)
            num_junctions.append(len(wn.junction_name_list))
            num_pipes.append(len(wn.pipe_name_list))
            time_steps.append(idx)
        except Exception as e:
            print(f"Failed to load {inp_file} in {scenario}: {e}")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, num_junctions, label='Junctions', marker='o')
    plt.plot(time_steps, num_pipes, label='Pipes', marker='x')
    plt.title(f"Network Growth Over Time: {scenario}")
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

