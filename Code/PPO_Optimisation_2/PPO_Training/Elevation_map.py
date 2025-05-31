
"""

This function is used to generated an elevation map to extract values from for the network.

NOT CURRENTLY IN USE

"""

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

import numpy as np
import matplotlib.pyplot as plt

def generate_elevation_map(area_size=(100, 100), elevation_range=(0, 100), 
                           num_peaks=10, landscape_type='flat', seed=None):
    """
    Generate an elevation map using Gaussian peaks.
    
    Parameters:
    - area_size: Tuple[int, int], the (width, height) of the area (km)
    - elevation_range: Tuple[float, float], (min_elevation, max_elevation) (m)
    - num_peaks: int, number of elevation peaks
    - landscape_type: str, either 'flat' or 'hilly'
    - seed: Optional[int], for reproducibility
    
    Returns:
    - elevation_map: 2D numpy array of shape (height, width)
    - peak_data: List of tuples (x, y, z) for each peak
    """
    if seed is not None:
        np.random.seed(seed)
    
    width, height = area_size
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)
    elevation_map = np.zeros_like(X)

    # Scale sigma values according to area size
    avg_dimension = (width + height) / 2
    sigma_scale = avg_dimension / 100  # Adjust this scale factor as needed

    # Define Gaussian parameters based on landscape type
    if landscape_type == 'flat':
        sigma_range = (20 * sigma_scale, 40 * sigma_scale)
        amp_range = (10, 30)
    elif landscape_type == 'hilly':
        sigma_range = (5 * sigma_scale, 15 * sigma_scale)
        amp_range = (30, 70)
    else:
        raise ValueError("landscape_type must be 'flat' or 'hilly'")

    peak_data = []

    for _ in range(num_peaks):
        x0 = np.random.uniform(0, width)
        y0 = np.random.uniform(0, height)
        sigma = np.random.uniform(*sigma_range)
        amplitude = np.random.uniform(*amp_range)

        Z = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        elevation_map += Z
        peak_data.append((x0, y0, amplitude))

    # Normalize elevation to match the desired elevation_range
    min_elev, max_elev = elevation_range
    elevation_map = (elevation_map - elevation_map.min()) / (elevation_map.max() - elevation_map.min())
    elevation_map = elevation_map * (max_elev - min_elev) + min_elev

    return elevation_map, peak_data

def plot_elevation_map(elevation_map, peaks=None):
    """
    Plots the elevation map using a color gradient.

    Parameters:
    - elevation_map: Elevation map to be plotted.
    """

    plt.contourf(elevation_map, cmap='terrain', alpha = 0.5, levels = 20)
    plt.colorbar(label='Elevation')
    # Add positions of peaks
    if peaks is not None:
        for peak in peaks:
            plt.plot(peak[0], peak[1], 'ro', markersize=8)
    plt.title('Elevation Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

if __name__ == "__main__":
    # Example usage
    elevation_map, peaks = generate_elevation_map(area_size=(1000, 1000), 
                                                  elevation_range=(0, 100), 
                                                  num_peaks=2, 
                                                  landscape_type='flat')
    plot_elevation_map(elevation_map, peaks)