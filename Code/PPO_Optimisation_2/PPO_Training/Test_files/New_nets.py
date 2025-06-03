
"""

Test that new networks have the correct number of pipes in each stepo for each scenario

"""

import os
import matplotlib.pyplot as plt
from wntr.network import WaterNetworkModel

# --- Configuration ---
networks_folder = 'Modified_nets'
scenarios = [s for s in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, s))]

sprawling_scenarios = [s for s in scenarios if 'sprawling' in s]

def plot_junctions_and_pipes(scenario, num_junctions, num_pipes, time_steps):
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

def extract_step_number(filename):
    """Extract the step number from a filename like 'Step_10.inp'"""
    try:
        return int(filename.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return 0  # Default value for files that don't match the pattern

if __name__ == "__main__":
    # --- Configuration ---
    networks_folder = 'Modified_nets'
    scenarios = [s for s in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, s))]

    sprawling_scenarios = [s for s in scenarios if 'sprawling' in s]

    # --- Iterate through each scenario ---
    for scenario in sprawling_scenarios:
        scenario_path = os.path.join(networks_folder, scenario)
        # Get all inp files
        inp_files = [f for f in os.listdir(scenario_path) if f.endswith('.inp')]
        # Sort by the actual step number, not alphabetically
        inp_files.sort(key=extract_step_number)

        num_junctions = []
        num_pipes = []
        time_steps = []

        for inp_file in inp_files:
            full_path = os.path.join(scenario_path, inp_file)
            try:
                # Extract the step number from the filename
                step_num = extract_step_number(inp_file)
                
                wn = WaterNetworkModel(full_path)
                num_junctions.append(len(wn.junction_name_list))
                num_pipes.append(len(wn.pipe_name_list))
                time_steps.append(step_num)
            except Exception as e:
                print(f"Failed to load {inp_file} in {scenario}: {e}")

        # --- Plot ---
        if num_junctions and num_pipes:
            # Sort data by time steps before plotting
            sorted_data = sorted(zip(time_steps, num_junctions, num_pipes))
            time_steps = [item[0] for item in sorted_data]
            num_junctions = [item[1] for item in sorted_data]
            num_pipes = [item[2] for item in sorted_data]
            
            plot_junctions_and_pipes(scenario, num_junctions, num_pipes, time_steps)
        else:
            print(f"No valid data found for scenario: {scenario}")
