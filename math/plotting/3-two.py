#!/usr/bin/env python3
"""
This module contains a function that plots the exponential decay
of two different radioactive elements on the same graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots the decay of C-14 and Ra-226.
    Includes axis labels, title, a legend, and specific line styles.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 as a dashed red line with label for legend
    plt.plot(x, y1, 'r--', label='C-14')

    # Plot y2 as a solid green line with label for legend
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Set axis labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Set axis ranges
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
