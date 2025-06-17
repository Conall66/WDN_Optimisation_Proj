
"""

In this script, we configure the setting for a hydraulic simulation and extract key performance metrics from the results

"""

import wntr
from wntr.network import WaterNetworkModel
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import gc
import time

# def run_epanet_sim(wn):
#     """
#     Run the hydraulic simulation using the WNTR library.
    
#     Parameters:
#     wn (WaterNetworkModel): The water network model to simulate.
    
#     Returns:
#     tuple: A tuple containing the hydraulic results and quality results.
#     """

#     # Set up the hydraulic simulation options
#     wn.options.time.hydraulic_timestep = 3600
#     wn.options.time.pattern_timestep = 3600
#     wn.options.time.report_timestep = 3600
#     wn.options.hydraulic.inpfile_units = 'CMH'
#     wn.options.hydraulic.accuracy = 0.01
#     wn.options.hydraulic.trials = 100
#     wn.options.hydraulic.headloss = 'H-W'
#     wn.options.hydraulic.demand_model = 'DDA'
#     wn.options.energy.global_efficiency = 100.0
#     wn.options.energy.global_price = 0.26

#     sim = wntr.sim.EpanetSimulator(wn)
#     results = sim.run_sim()
#     return results

def run_epanet_sim(wn, static=False):
    """
    Run EPANET simulation on the water network model in a process-safe temporary directory
    with a robust cleanup mechanism to avoid file lock errors in parallel execution.
    """
    # --- Simulation options setup (no changes here) ---
    if static:
        wn.options.time.duration = 0
    else:
        wn.options.time.duration = 24 * 3600

    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.inpfile_units = 'CMH'
    wn.options.hydraulic.accuracy = 0.01
    wn.options.hydraulic.trials = 100
    wn.options.hydraulic.headloss = 'H-W'
    wn.options.hydraulic.demand_model = 'DDA'
    wn.options.energy.global_efficiency = 100.0
    wn.options.energy.global_price = 0.26

    # --- MODIFICATION FOR ROBUST PARALLEL EXECUTION ---

    # 1. Store the original working directory.
    original_cwd = os.getcwd()
    
    # 2. Manually create a temporary directory instead of using a 'with' block.
    # This gives us full control over the cleanup process.
    temp_dir_path = tempfile.mkdtemp(prefix="wntr_sim_")
    
    results = None
    
    try:
        # 3. Change into the isolated directory to run the simulation.
        os.chdir(temp_dir_path)
        
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

    except Exception as e:
        print(f"ERROR: Exception during EPANET simulation in temp dir {temp_dir_path}: {e}")
        # The finally block will still run to perform cleanup.
        raise  # Re-raise the exception so the calling environment knows about the failure.

    finally:
        # 4. CRITICAL: Always change the CWD back to the original path first.
        os.chdir(original_cwd)
        gc.collect() # Force garbage collection to release any file handles.
        
        # 5. Implement a robust cleanup with a retry loop.
        for i in range(5):  # Try to clean up 5 times
            try:
                shutil.rmtree(temp_dir_path)
                # If deletion is successful, break the loop
                break
            except PermissionError:
                # This catches the exact [WinError 32] you are seeing.
                # print(f"Cleanup attempt {i+1} failed for {temp_dir_path}. Retrying in 0.1s...")
                time.sleep(0.1)

            except Exception as e:
                print(f"Warning: An unexpected error occurred during cleanup of {temp_dir_path}: {e}")
                break
        else:
            # This 'else' block runs only if the 'for' loop completes without a 'break'.
            # This means all cleanup attempts failed.
            print(f"Warning: Could not clean up temporary directory {temp_dir_path}. It may need to be deleted manually.")
    
    # The rest of the function remains the same.
    return results
