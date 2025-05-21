
"""

In this file, we choose networks from the imported networks to add to Networks subfolder

"""

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import shutil

# For each network in the new Networks folder, we will add to a subplot and visualise
def visualise_networks_in_subplots(networks_folder):
    # Get a list of all .inp files in the networks folder
    inp_files = [f for f in os.listdir(networks_folder) if f.endswith('.inp')]
    
    # Create a figure with subplots
    num_files = len(inp_files)
    cols = 3
    rows = (num_files // cols) + (num_files % cols > 0)
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    
    for i, inp_file in enumerate(inp_files):
        try:
            # Load the network
            file_path = os.path.join(networks_folder, inp_file)
            print(f"Loading {inp_file}...")
            wn = wntr.network.WaterNetworkModel(file_path)
            
            # Visualise the network
            wntr.graphics.plot_network(wn, ax=axs[i])
            axs[i].set_title(inp_file)
        except Exception as e:
            print(f"Error loading {inp_file}: {str(e)}")
            axs[i].text(0.5, 0.5, f"Error loading {inp_file}", 
                        ha='center', va='center', transform=axs[i].transAxes, color='red')
            axs[i].set_title(inp_file)
    
    # Hide any unused subplots
    for j in range(num_files, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def fix_inp_file_units(file_path):
    """Fix SI units in EPANET .inp files by replacing with LPS (liters per second)"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace 'SI' units with 'LPS' (liters per second) - a valid EPANET unit
    if ' SI\n' in content:
        content = content.replace(' SI\n', ' LPS\n')
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed units in {os.path.basename(file_path)}")
        return True
    return False

def check_inp_file(file_path):
    """Check an INP file for common issues and try to fix them"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for specific issues in the JEP5-13.inp file
        if os.path.basename(file_path) == "JEP5-13.inp":
            # Examine the content of this specific file
            print(f"Examining {os.path.basename(file_path)} for issues...")
            
            # Try loading the model to see specific errors
            try:
                wn = wntr.network.WaterNetworkModel(file_path)
            except Exception as e:
                print(f"Specific error: {str(e)}")
        
        # Common EPANET fixes for any file
        # fixed = fix_inp_file_units(file_path)
        """Changing file units without scaling can cause problems in the simulation"""
        return True
            
    except Exception as e:
        print(f"Error checking {os.path.basename(file_path)}: {str(e)}")
        return False
    
def visualise_networks_by_folder(networks_folder):
    """Visualize networks grouped by their source folders"""
    # Get a list of all subfolders
    folders = [f for f in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, f))]
    
    for folder in folders:
        folder_path = os.path.join(networks_folder, folder)
        inp_files = [f for f in os.listdir(folder_path) if f.endswith('.inp')]
        
        if not inp_files:
            print(f"No .inp files found in {folder}")
            continue
            
        print(f"\nVisualizing networks from folder: {folder}")
        
        # Create a figure with subplots
        num_files = len(inp_files)
        cols = 3
        rows = (num_files // cols) + (num_files % cols > 0)
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle(f"Networks from {folder}", fontsize=16)
        
        # Handle the case of a single plot
        if num_files == 1:
            axs = np.array([axs])
        
        # Flatten the axs array for easier indexing
        axs = axs.flatten()
        
        for i, inp_file in enumerate(inp_files):
            try:
                # Load the network
                file_path = os.path.join(folder_path, inp_file)
                print(f"Loading {inp_file}...")
                wn = wntr.network.WaterNetworkModel(file_path)
                
                # Visualise the network
                wntr.graphics.plot_network(wn, ax=axs[i])
                axs[i].set_title(inp_file)
            except Exception as e:
                print(f"Error loading {inp_file}: {str(e)}")
                axs[i].text(0.5, 0.5, f"Error loading {inp_file}", 
                            ha='center', va='center', transform=axs[i].transAxes, color='red')
                axs[i].set_title(inp_file)
        
        # Hide any unused subplots
        for j in range(num_files, len(axs)):
            axs[j].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.show()

def copy_networks_from_all_folders(base_source_path, destination_base_path):
    """Copy networks from all folders in epanet-tests to corresponding subfolders in destination"""
    # Get all subfolders in the source directory
    if not os.path.exists(base_source_path):
        print(f"Source path does not exist: {base_source_path}")
        return

    # List all subfolders in the source directory
    subfolders = [f for f in os.listdir(base_source_path) if os.path.isdir(os.path.join(base_source_path, f))]
    
    if not subfolders:
        print(f"No subfolders found in {base_source_path}")
        return
    
    print(f"Found {len(subfolders)} subfolders: {subfolders}")
    
    for subfolder in subfolders:
        source_folder = os.path.join(base_source_path, subfolder)
        destination_folder = os.path.join(destination_base_path, subfolder)
        
        # Create destination subfolder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")
        
        # Copy all .inp files from this subfolder
        count = 0
        for file in os.listdir(source_folder):
            if file.endswith(".inp"):
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(destination_folder, file)
                shutil.copy(source_file, destination_file)
                count += 1
        
        print(f"Copied {count} .inp files from {subfolder} to {destination_folder}")
        
        # Fix any issues in the copied files
        for file in os.listdir(destination_folder):
            if file.endswith(".inp"):
                file_path = os.path.join(destination_folder, file)
                check_inp_file(file_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Base paths
    epanet_tests_path = os.path.join(script_dir, "Imported_networks", "epanet-example-networks", "epanet-tests")
    networks_base_path = os.path.join(script_dir, "Networks")
    
    # Ensure the networks base directory exists
    if not os.path.exists(networks_base_path):
        os.makedirs(networks_base_path)
    
    # Copy networks from all subfolders
    copy_networks_from_all_folders(epanet_tests_path, networks_base_path)
    
    # Create lists of working and problematic networks by folder
    print("\nTesting all networks...")
    
    working_networks = {}
    problematic_networks = {}
    
    for folder in os.listdir(networks_base_path):
        folder_path = os.path.join(networks_base_path, folder)
        
        if not os.path.isdir(folder_path):
            continue
            
        working_networks[folder] = []
        problematic_networks[folder] = []
        
        for file in os.listdir(folder_path):
            if file.endswith(".inp"):
                file_path = os.path.join(folder_path, file)
                try:
                    wn = wntr.network.WaterNetworkModel(file_path)
                    working_networks[folder].append(file)
                except Exception as e:
                    print(f"Network {file} in {folder} has errors: {str(e)}")
                    problematic_networks[folder].append(file)
    
    # Print summary
    print("\n--- Network Test Summary ---")
    for folder in working_networks:
        print(f"\nFolder: {folder}")
        print(f"  Working networks: {len(working_networks[folder])}")
        print(f"  Problematic networks: {len(problematic_networks[folder])}")
        
        if problematic_networks[folder]:
            print(f"  Problematic networks in {folder}:")
            for file in problematic_networks[folder]:
                print(f"    - {file}")
    
    # Visualize networks by folder
    visualise_networks_by_folder(networks_base_path)

"""

Files that are not working or showed no interesting topology or were too large were forceably removed from the networks folder.

"""