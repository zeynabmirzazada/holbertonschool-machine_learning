#!/usr/bin/env python3
"""
This module contains a function that plots a stacked bar graph
representing the quantity of fruit possessed by different people.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph with specific colors, labels, and axis limits.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5

    # Initialize the bottom of the bars at 0
    bottom_val = np.zeros(3)

    # Loop through each row (fruit type) to stack them
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=width, bottom=bottom_val,
                color=colors[i], label=fruit_names[i])
        # Update the bottom value for the next fruit in the stack
        bottom_val += fruit[i]

    # Set the y-axis label and range
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Set the title and legend
    plt.title('Number of Fruit per Person')
    plt.legend()

    # Display the plot
    plt.show()
