#!/usr/bin/env python3
"""
This module contains a function that plots a histogram of student grades.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student scores for Project A.
    Bins are set every 10 units and bars are outlined in black.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100
    bins = range(0, 101, 10)

    # Plot the histogram
    # edgecolor='black' adds the requested outline
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Display the plot
    plt.show()
