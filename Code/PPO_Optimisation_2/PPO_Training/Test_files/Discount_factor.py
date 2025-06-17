
"""

Plow how different discount factors change over time in the network

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def plot_discount_factors(discount_factors, time_steps, save_path = None):
    """
    Plots the discount factors over time.
    
    Parameters:
    - discount_factors: List of discount factors for each time step.
    - time_steps: Number of time steps.
    """

    plt.figure(figsize=(10, 6))
    for i in range(len(discount_factors)):
        cum_discounts = []
        for j in range(time_steps):
            cum_discounts.append(discount_factors[i] ** (j + 1))
        plt.plot(range(time_steps), cum_discounts)

    legend_labels = []
    for i, df in enumerate(discount_factors):
        legend_labels.append(f'\u03B3 = {df}') # Gamma short code

    plt.title('Discount Factors Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Discount Factor')
    plt.legend(legend_labels, loc='upper right')
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    discount_factors = [0.99, 0.95, 0.90, 0.85]  # Example discount factors
    time_steps = 50  # Example number of time steps
    script = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script)
    plot_dir = os.path.join(parent_dir, 'Plots', 'Tests')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'discount_factors_over_time.png')

    plot_discount_factors(discount_factors, time_steps, save_path)