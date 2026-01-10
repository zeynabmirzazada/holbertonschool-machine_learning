#!/usr/bin/env python3
"""
This module contains a function that plots a simple line graph
representing a cubic function.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y as a solid red line.
    The x-axis ranges from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Plot y as a solid red line
    # If x values are not provided, matplotlib uses the index (0-10)
    plt.plot(y, 'r-')

    # Set the x-axis range from 0 to 10
    plt.xlim(0, 10)

    # Display the plot
    plt.show()
