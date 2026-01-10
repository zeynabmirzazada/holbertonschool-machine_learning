#!/usr/bin/env python3
"""
This module contains a function that plots the exponential decay of Carbon-14
using a logarithmic scale on the y-axis.
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the fraction of C-14 remaining over time.
    The y-axis is logarithmically scaled to show the linear relationship
    of exponential decay in log-space.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the line graph
    plt.plot(x, y)

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Set the x-axis range
    plt.xlim(0, 28650)

    # Display the plot
    plt.show()
