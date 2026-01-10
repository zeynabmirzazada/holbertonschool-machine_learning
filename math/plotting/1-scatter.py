#!/usr/bin/env python3
"""
This module contains a function that plots a scatter plot of
height versus weight data.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of Men's Height vs Weight.
    The points are magenta, axes are labeled, and a title is included.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the scatter points in magenta
    plt.scatter(x, y, c='m')

    # Add labels and title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
