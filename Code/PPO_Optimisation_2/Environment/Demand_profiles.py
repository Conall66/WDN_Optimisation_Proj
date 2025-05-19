
"""

In this file, we project the demand profiles of the network for residential and commercial use by day and seasonas a function of the population growth. Growth figures were taken from predictions for sub-Saharan Africa, and demand was determinde accordingly. Demand is projected for 25 years until 2050, and measured in literes per second.

"""

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import math
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(5)

def generate_sample_growth(start_year = 2025, end_year = 2050, start_population = 20000, end_population = 50000, num_samples = 100):
    """
    Generate a sample of population growth values between start_year and end_year following an S-curve pattern.
    
    Parameters:
    start_year (int): The starting year for the population growth.
    end_year (int): The ending year for the population growth.
    start_population (int): The starting population.
    end_population (int): The ending population.
    num_samples (int): The number of samples to generate.
    
    Returns:
    list: A list of population growth values.
    """
    
    quarters = np.linspace(start_year, end_year, num_samples)
    populations = []
    for Q in quarters:
        # S-curve formula: P(t) = P0 + (P1 - P0) / (1 + e^(-k(t - t0)))
        k = 0.1  # steepness of the curve
        t0 = (start_year + end_year) / 2  # midpoint of the curve
        population = start_population + (end_population - start_population) / (1 + np.exp(-k * (Q - t0)))
        populations.append(population)
    
    return quarters, populations

# Function for generating demand profiles for residential and commercial use
def generate_demand_profiles(population_growth, uncertainty = None):
    """
    Generate demand curve for residential and commercial use based on population growth. This provides the total demand by hour/quarter across the population (population per household for example then can be extracted through dividing the resdeintial demand by the number of residential nodes)
    
    Parameters:
    conusmer (str): The type of consumer (e.g., 'residential', 'commercial').
    population_growth (list): A list of population growth values by quarter.
    uncertainty_factor (float): The factor to apply to the demand values for uncertainty.
    
    Returns:
    dict: A dictionary containing demand profiles for residential and commercial use.
    """

    # Create a dictionary for daily expected residential demand by quarter

    # Separate the population growth into quarters
    quarters = []
    for i in range(len(population_growth)):
        year = int(i / 4) + 2025
        quarter = (i % 4) + 1
        quarters.append(f"Y{year}Q{quarter}")

    # Residential demand values
    hours = np.arange(0, 24, 0.25)

    res_base_demand = (
        1.5 * np.exp(-0.5 * (hours - 7)**2) +  # Morning peak
        2.0 * np.exp(-0.5 * (hours - 19)**2) + # Evening peak
        0.5                                    # Base load
    )

    # Add a small degree of noise to the residential demand
    noise = np.random.normal(0, 0.1, len(res_base_demand))
    res_base_demand += noise
    res_base_demand = np.clip(res_base_demand, 0, None)  # Ensure no negative values

    # Simulate commercial: business hours plateau
    com_base_demand = (
        3.0 * (np.heaviside(hours - 8, 1) - np.heaviside(hours - 18, 1)) + # Flat peak
        0.2                                                               # Idle load
    )

    # Add a small degree of noise to the commercial demand
    noise = np.random.normal(0, 0.1, len(com_base_demand))
    com_base_demand += noise
    com_base_demand = np.clip(com_base_demand, 0, None)  # Ensure no negative values

    season_map = {
        'Q1': 'Winter',
        'Q2': 'Spring',
        'Q3': 'Summer',
        'Q4': 'Autumn'
    }

    # Model seasonal variation
    seasonal_variation = {
        'Q1': 0.8,  # Winter
        'Q2': 1.2,  # Spring
        'Q3': 1.5,  # Summer
        'Q4': 1.0   # Autumn
    }

    # Assume uncertainty increases linearly with time, starting from 0 and ending at 0.8 (80% uncertainty)
    if uncertainty is None:
        uncertainty = np.linspace(0, 0.8, len(population_growth))

    # Initialize demand dictionaries
    residential_demand = {}
    commercial_demand = {}
    
    # Generate demand for each quarter
    for i, pop in enumerate(population_growth):
        year = int(i / 4) + 2025
        quarter_num = (i % 4) + 1
        quarter_key = f"Y{year}Q{quarter_num}"
        quarter_id = f"Q{quarter_num}"
        
        # Apply seasonal variation and population scaling
        # Assuming demand scales linearly with population, normalized to initial population
        pop_factor = pop / population_growth[0]
        seasonal_factor = seasonal_variation[quarter_id]
        
        # Calculate scaled demand patterns
        res_hourly_demand = res_base_demand * pop_factor * seasonal_factor
        com_hourly_demand = com_base_demand * pop_factor * seasonal_factor
        
        # Store in dictionaries with proper structure
        residential_demand[quarter_key] = {
            "Season": season_map[quarter_id],
            "Demand": res_hourly_demand.tolist(),
            "Uncertainty": uncertainty[i]
        }
        
        commercial_demand[quarter_key] = {
            "Season": season_map[quarter_id],
            "Demand": com_hourly_demand.tolist(),
            "Uncertainty": uncertainty[i]
        }
    
    return residential_demand, commercial_demand


if __name__ == "__main__":
    
    # Create an arbitrary populaiton growth function, modelling as an S-curve starting at 20,000 people and growing to 50,000 over the 25 years

    start_year = 2025
    end_year = 2050
    start_population = 20000
    end_population = 50000
    num_samples = 25
    years, populations = generate_sample_growth(start_year, end_year, start_population, end_population, num_samples)
    residential_demand, commercial_demand = generate_demand_profiles(populations, uncertainty = None)

    # Print the demand profiles
    for quarter, demand in residential_demand.items():
        print(f"{quarter}: Residential Demand: {demand['Demand']}, Uncertainty: {demand['Uncertainty']}")
    for quarter, demand in commercial_demand.items():
        print(f"{quarter}: Commercial Demand: {demand['Demand']}, Uncertainty: {demand['Uncertainty']}")

    # Plot winter 2025 and summer 2025, winter 2050 and summer 2050
    quarters_to_plot = ['Y2025Q1', 'Y2025Q3', 'Y2050Q1', 'Y2050Q3']
    hours = np.arange(0, 24, 0.25)  # Create time axis for plotting
    
    plt.figure(figsize=(12, 6))
    for quarter in quarters_to_plot:
        if quarter in residential_demand:
            res_demand = residential_demand[quarter]['Demand']
            res_uncertainty = residential_demand[quarter]['Uncertainty']
            plt.plot(hours, res_demand, label=f"Residential {quarter}")
            plt.fill_between(hours, 
                            np.array(res_demand)*(1-res_uncertainty), 
                            np.array(res_demand)*(1+res_uncertainty), 
                            alpha=0.2)
        
        if quarter in commercial_demand:
            com_demand = commercial_demand[quarter]['Demand']
            com_uncertainty = commercial_demand[quarter]['Uncertainty']
            plt.plot(hours, com_demand, label=f"Commercial {quarter}")
            plt.fill_between(hours, 
                            np.array(com_demand)*(1-com_uncertainty), 
                            np.array(com_demand)*(1+com_uncertainty), 
                            alpha=0.2)
    
    plt.title("Demand Profiles")
    plt.xlabel("Time (hours)")
    plt.ylabel("Demand (L/s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



