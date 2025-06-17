
import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import wntr # For creating a dummy .inp file
import sys

script = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script)
# Add the parent directory to the Python path to import GA_Alt_Approach and its dependencies
sys.path.append(parent_dir)


# Import necessary components from your GA_Alt_Approach.py and other project files
# Ensure GA_Alt_Approach.py and its dependencies are in the Python path
try:
    from GA_Alt_Approach import (
        run_ga_drl_comparison,
        PIPES_CONFIG,
        LABOUR_COST,
        NETWORKS_FOLDER_PATH,
        plot_comparison_results # Optionally, to visualize test results
    )
    # These are imported by GA_Alt_Approach, but good to be aware of them:
    # from Hydraulic_Model import run_epanet_simulation, evaluate_network_performance
    # from Reward import calculate_reward, compute_total_cost
    # from PPO_Environment import WNTRGymEnv
    # from Actor_Critic_Nets2 import GraphPPOAgent
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure GA_Alt_Approach.py and all its dependencies (Hydraulic_Model.py, Reward.py, etc.) are in the same directory or accessible in your Python path.")
    exit()

def create_dummy_drl_model(agents_dir, model_name="trained_gnn_ppo_wn_dummy_test.zip"):
    """Creates a dummy DRL model file if none exists, to allow GA test to run."""
    dummy_model_path = os.path.join(agents_dir, model_name)
    if not os.path.exists(dummy_model_path):
        try:
            # A real PPO model save includes a zip file. We'll just create an empty file.
            with open(dummy_model_path, 'w') as f:
                f.write("This is a dummy DRL model file for GA testing.")
            print(f"Created dummy DRL model file: {dummy_model_path}")
            return dummy_model_path
        except Exception as e:
            print(f"Could not create dummy DRL model file: {e}")
    elif os.path.exists(dummy_model_path):
        print(f"Using existing dummy/placeholder DRL model: {dummy_model_path}")
        return dummy_model_path
    return None

def create_dummy_inp_file(base_path, scenario_name, inp_filename="Step_0.inp"):
    """Creates a minimal dummy WNTR .inp file for testing if it doesn't exist."""
    scenario_dir = os.path.join(base_path, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    dummy_inp_path = os.path.join(scenario_dir, inp_filename)

    if not os.path.exists(dummy_inp_path):
        try:
            wn_dummy = wntr.network.WaterNetworkModel()
            # Add minimal components to make it a valid network
            wn_dummy.add_reservoir("R1", base_head=20.0, coordinates=(0,0))
            wn_dummy.add_junction("J1", base_demand=0.05, elevation=10.0, coordinates=(100,0))
            wn_dummy.add_junction("J2", base_demand=0.05, elevation=5.0, coordinates=(200,0))
            
            # Use a diameter that exists in PIPES_CONFIG
            valid_diameter = PIPES_CONFIG['Pipe_1']['diameter'] if PIPES_CONFIG else 0.3048

            wn_dummy.add_pipe("P1", "R1", "J1", length=100.0, diameter=valid_diameter, roughness=100, minor_loss=0.0)
            wn_dummy.add_pipe("P2", "J1", "J2", length=100.0, diameter=valid_diameter, roughness=100, minor_loss=0.0)
            
            wn_dummy.options.time.duration = 3600 # 1 hour simulation
            wn_dummy.options.time.hydraulic_timestep = 3600
            wn_dummy.options.hydraulic.headloss = 'H-W' # Match expected headloss formula
            wn_dummy.options.hydraulic.units = 'CMH' # Match expected units

            wn_dummy.write_inpfile(dummy_inp_path)
            print(f"Created dummy .inp file: {dummy_inp_path}")
        except Exception as e:
            print(f"Could not create dummy .inp file {dummy_inp_path}: {e}")
            print("Please ensure a valid .inp file exists for the test scenario or check PIPES_CONFIG.")
    return dummy_inp_path

def run_ga_test():
    """Main function to run the GA functionality test."""
    print("--- Starting GA Functionality Test ---")

    # --- Test Configuration ---
    # 1. DRL Model Path (required by run_ga_drl_comparison)
    agents_dir = "agents"
    os.makedirs(agents_dir, exist_ok=True) # Ensure agents directory exists
    
    # Try to find any existing DRL model to prevent `run_ga_drl_comparison` from failing early
    existing_drl_models = [
        os.path.join(agents_dir, f)
        for f in os.listdir(agents_dir)
        if f.startswith("trained_gnn_ppo_wn_") and f.endswith(".zip")
    ]
    if existing_drl_models:
        drl_model_path_for_test = max(existing_drl_models, key=os.path.getctime)
        print(f"Using existing DRL model for comparison part: {drl_model_path_for_test}")
    else:
        # If no real DRL model, create/use a dummy one so the GA test part can proceed
        drl_model_path_for_test = create_dummy_drl_model(agents_dir)
        if not drl_model_path_for_test:
            print("Error: Could not find or create a DRL model path. The GA comparison script might fail.")
            print("The GA test focuses on GA functionality, but the combined script needs a DRL path.")
            # Depending on how run_ga_drl_comparison handles a missing DRL model,
            # you might need to ensure a file (even empty) exists at the expected path.
            # For now, we'll proceed if a dummy was created.
            if not drl_model_path_for_test: # If still None
                 print("Exiting test as no DRL model path could be established.")
                 return


    # 2. Scenario for GA Testing
    #    The GA in run_ga_drl_comparison optimizes the *final* .inp of this scenario.
    #    Let's use a simple, predictable scenario.
    #    The `NETWORKS_FOLDER_PATH` is imported from GA_Alt_Approach
    test_scenario_name = 'anytown_simple_test' # A specific name for our test scenario
    scenarios_to_test_ga_on = [test_scenario_name]
    
    # Create a dummy .inp file for this test scenario if it doesn't exist
    # The GA optimizes the *last* .inp file in the scenario folder. For simplicity,
    # we'll create just one, e.g., "Step_0.inp" or "final_state.inp".
    # GA_Alt_Approach expects a folder per scenario.
    create_dummy_inp_file(NETWORKS_FOLDER_PATH, test_scenario_name, inp_filename="final_state.inp")
    print(f"Testing GA on scenario: {test_scenario_name} (using {NETWORKS_FOLDER_PATH}/{test_scenario_name}/final_state.inp)")

    # 3. GA Parameters for a Quick Test
    test_ga_generations = 5       # Very small number of generations
    test_ga_population_size = 8   # Very small population
    test_ga_mutation_rate = 25      # Higher mutation for small pop/gens to encourage diversity

    print(f"GA Parameters: Generations={test_ga_generations}, Population={test_ga_population_size}, Mutation={test_ga_mutation_rate}%")

    # --- Run the GA-focused part of the comparison ---
    start_test_time = time.time()
    
    # Ensure results directories exist for the main script's plotting/saving
    os.makedirs("Plots", exist_ok=True)
    os.makedirs(os.path.join("Plots", "GA_DRL_Comparison_Charts"), exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    comparison_df = run_ga_drl_comparison(
        drl_model_path=drl_model_path_for_test,
        scenarios_to_compare=scenarios_to_test_ga_on,
        ga_generations=test_ga_generations,
        ga_pop_size=test_ga_population_size,
        ga_mutation_percent=test_ga_mutation_rate
    )
    
    test_duration = time.time() - start_test_time
    print(f"\nGA Test Run (including DRL eval part if model was found) completed in {test_duration:.2f} seconds.")

    # --- Display Key GA Results ---
    if comparison_df is not None and not comparison_df.empty:
        ga_results = comparison_df[comparison_df['Method'] == 'GA']
        if not ga_results.empty:
            print("\n--- GA Test Run Results (from comparison_df) ---")
            print(ga_results.to_string())
            
            # To see the actual best chromosome, you would typically need to modify
            # run_ga_drl_comparison to return/log ga_instance.best_solution()[0]
            # For this test, we focus on whether it runs and produces fitness.
            print("\nTo see the best pipe diameters chosen by GA, you might need to:")
            print("1. Add print statements within the GA fitness function or at the end of ga_instance.run().")
            print("2. Modify 'run_ga_drl_comparison' to return the best chromosome.")

            # Optionally, generate and show the plots for this test run
            # test_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_test")
            # plot_comparison_results(comparison_df, test_timestamp_str)
            # plt.show() # Uncomment if you want to see plots immediately
        else:
            print("No GA-specific results found in the comparison DataFrame. The GA run might have failed or produced no valid solutions.")
    else:
        print("Test run did not produce a comparison DataFrame, or it was empty. Check for errors during the run.")

    print("\n--- GA Functionality Test Finished ---")

if __name__ == "__main__":
    run_ga_test()
