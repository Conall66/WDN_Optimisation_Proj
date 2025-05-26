
"""

In this file, we generate demand forecasts to capture the 3 SSP scenarios of rapid, expeected and slow growth. We can then take a water distribution network with existing demands and update these demands in accordance with the forecasts.

"""

# Import libraries
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

# Function to generate demand curves - this creates S curve profiles with final increase values of 50%, 65% and 80% respectively.
def gen_curve(x, final_val, midpoint, steepness = 0.5):
    """
    Takes an array x and generates a sigmoid curve with a final value of final_val and steepness.
    """
    base_curve = 1 + (final_val - 1) / (1 + np.exp(-steepness * (x - midpoint)))
    return base_curve

def gen_seasonal_variation(demand_curves, summer_amplitude = 0.05, winter_ratio = 0.3):
    """
    Adds seasonal variation to demand curves, alternating between summer and winter.
    
    Args:
        demand_curves (list): List of demand curves to apply seasonal variation to
        amplitude (float): Maximum percentage variation (0.15 = 15% variation)
    
    Returns:
        list: Demand curves with seasonal variation applied
    """
    seasonal_curves = []
    for curve in demand_curves:
        # Create a seasonal pattern that alternates every 2 steps
        steps = len(curve)
        seasonal_pattern = np.zeros(steps)
        
        # Calculate winter amplitude based on summer amplitude and the ratio
        winter_amplitude = summer_amplitude * winter_ratio
        
        # Generate alternating pattern (every 2 steps)
        for i in range(steps):
            if (i // 2) % 2 == 0:  # Summer (higher demand)
                seasonal_pattern[i] = summer_amplitude
            else:  # Winter (lower demand)
                seasonal_pattern[i] = -winter_amplitude
        
        # Apply seasonal variation to the base curve
        seasonal_curve = curve * (1 + seasonal_pattern)
        seasonal_curves.append(seasonal_curve)
    
    return seasonal_curves

def smooth_curve(curve, window_size=5):
    """
    Smooths a curve using a simple moving average.
    
    Args:
        curve (np.array): The input curve to be smoothed
        window_size (int): The size of the moving average window
    
    Returns:
        np.array: The smoothed curve
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    return np.convolve(curve, np.ones(window_size) / window_size, mode='valid')

def add_noise(curve, noise_level=0.05):
    """
    Adds random noise to a curve.
    
    Args:
        curve (np.array): The input curve to which noise will be added
        noise_level (float): The standard deviation of the noise to be added
    
    Returns:
        np.array: The curve with added noise
    """
    noise = np.random.normal(0, noise_level, len(curve))
    return curve + noise

def generate_demand_curves(time_steps, plot=False):
    # Use padding years for smoothing, but only plot 2025-2050
    actual_start_year = 2025
    actual_end_year = 2050
    
    # Add padding for smoothing (2 years on each side)
    # padding = 4
    # start_year = actual_start_year - padding
    # end_year = actual_end_year + padding
    
    # Generate points with padding
    points = np.linspace(actual_start_year, actual_end_year, time_steps)
    
    # Generate demand curves for each scenario
    final_growth_vals = [1.5, 1.65, 1.8]  # Convert percentages to multipliers
    midpoint = actual_start_year + (actual_end_year - actual_start_year) / 2  # Midpoint of the curve
    colours = ['blue', 'orange', 'red']  # Colors for each curve

    # Extract demand curves for each of the scenarios
    demand_curves = [gen_curve(points, final_val, midpoint) for final_val in final_growth_vals]

    # Add seasonal variation
    seasonal_curves = gen_seasonal_variation(demand_curves)
    
    # Apply smoothing and noise to each curve
    processed_curves = []
    for curve in seasonal_curves:
        # First smooth the curve using 'same' mode to preserve length
        # smoothed = np.convolve(curve, np.ones(10) / 10, mode='same')
        # Then add noise to the smoothed curve
        noisy = add_noise(curve, noise_level=0.02)
        processed_curves.append(noisy)

    if plot:
        # Find indices for the actual range we want to plot (2025-2050)
        actual_range_mask = (points >= actual_start_year) & (points <= actual_end_year)
        plot_points = points[actual_range_mask]
        
        plt.figure(figsize=(10, 6))
        for i, curve in enumerate(processed_curves):
            # Extract only the points in our desired range
            plot_curve = curve[actual_range_mask]
            plt.plot(plot_points, plot_curve, label=f'{int((final_growth_vals[i]-1)*100)}% Growth', color=colours[i])

        # Legend inputs
        legend_labels = ['Low Growth (50%)', 'Expected Growth (65%)', 'High Growth (80%)']
        plt.title('Demand Growth Scenarios (2025-2050)')
        plt.xlabel('Year')
        plt.ylabel('Demand Multiplier')
        plt.legend(legend_labels, loc='upper left')
        plt.grid(True)
        # Save to plot folders
        if not os.path.exists('Plots'):
            os.makedirs('Plots')
        plt.savefig('Plots/demand_growth_scenarios.png')
        plt.show()

    # Return only the curves for the actual date range (2025-2050)
    actual_range_mask = (points >= actual_start_year) & (points <= actual_end_year)
    return [curve[actual_range_mask] for curve in processed_curves]

# Function to update network demands given demand curve and network

if __name__ == "__main__":
    # Generate demand curves
    time_steps = 50  # Number of time steps from 2025 to 2050
    demand_curves = generate_demand_curves(time_steps, plot = True)
    # Print the length of each of the demand curves
    for i, curve in enumerate(demand_curves):
        print(f"Demand curve {i+1} length: {len(curve)}")
