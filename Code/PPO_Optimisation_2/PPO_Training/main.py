
"""

This is the main file, in which initial networks ar extracted, and demand values updated according to forecasts. The PPO agent then takes as input these networks and learns to allocate pipe diameters to the network over time. 

"""

# Extract initial networks and run initial hydraulic analysis to set sa baseline for agent performance

# Visualise the initial networks with node pressure coloured

# Generate scenarios of each water network

# Create set of networks and store in separate folders to be accessed at by random by episode

# For each episode, select a random network from the set and run the agent to refine policy

# Extract agent performance by reward, run time and number of iterations

# Tabulate the performance of the agent by network topology, demand scenario, etc.

# Extract how pipe bursts/age affected agent performance

# Import a GA agent to compare performance with the PPO agent

# Extract a larger network from the initial networks folder and test both agents extracting performance metrics again