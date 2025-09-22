"""
This file compares the results of the random merging method with different group size limits
and saves the averaged data necessary to recreate the figure to CSV.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from algorithms.cluster_assignment_rand import cluster_assignment_rand
import test_utils as tu
from matplotlib import ticker

# NEW: imports for CSV export and timestamping
import csv
from datetime import datetime

"""HyperParameters"""
nu = 100 #number of robots # was 10
mu = 50 # number of tasks  # was 5
kappa = 3 # number of capabilities
L = 3 # maximum team size for a single task

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
}

# Define the environment size (min will always be 0)
max_x = 100
max_y = 100

"Test Parameters"
cluster_sizes = [3,4,5,6]
num_tests = 100
num_iterations = 100

# Define colors for different cluster sizes
colors = {
    3: '#1f77b4',  # blue
    4: '#ff7f0e',  # orange
    5: '#2ca02c',  # green
    6: '#d62728',  # red
    7: '#9467bd',  # purple
}

# Initialize a dictionary to store results for each cluster size
results = {size: [] for size in cluster_sizes}
time_results = {size: [] for size in cluster_sizes}

# Generate a discrete task requirements problem instance
if kappa == 3:
    robot_list, task_list = tu.generate_problem_3_capabilities(hypes, max_x, max_y)
elif kappa == 2:
    robot_list, task_list = tu.generate_problem_2_capabilities(hypes, max_x, max_y)
else:
    print("Error: kappa must be 2 or 3")

for test in range(num_tests):
    print(f"Test: {test+1}")
    
    for cluster_size in cluster_sizes:
        # Update hyperparameters for the current cluster size
        hypes['L_r'] = cluster_size
        hypes['L_t'] = cluster_size

        # Perform random iterative assignment
        total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
            robot_list, task_list, hypes, num_iterations
        )
        
        # Store the iteration rewards and times for this test and cluster size
        results[cluster_size].append(iteration_rewards)
        time_results[cluster_size].append((iteration_rewards, iteration_times))

# -----------------------------------------------
"""Compute Average Utility per Iteration with Different Cluster Sizes"""
# -----------------------------------------------

# Find the maximum number of iterations across all tests and cluster sizes
max_iter = max([max([len(rewards) for rewards in results[size]]) for size in cluster_sizes])
iterations = np.arange(1, max_iter + 1)

# Dictionary to store average data for each cluster size
avg_data = {size: np.zeros(max_iter) for size in cluster_sizes}

# Process and average the data for each cluster size
for size in cluster_sizes:
    # For each iteration, calculate the average reward across all tests
    for i in range(max_iter):
        # Get rewards for this iteration from all tests that have this iteration
        # If a test finished early, we carry forward the last known value (to match plotting logic)
        iter_rewards = [rewards[i] if i < len(rewards) else rewards[-1] for rewards in results[size]]
        avg_data[size][i] = np.mean(iter_rewards)

# -----------------------------------------------
""" Save averaged data to CSV (data required for regenerating the figure) """
# -----------------------------------------------

# Set up output directory and filename

# Save averaged CSV directly in the current working directory
avg_csv_path = f"avg_utility_per_iteration_kappa{kappa}_nu{nu}_mu{mu}_tests{num_tests}_its{num_iterations}.csv"

with open(avg_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Long-form header: iteration, cluster_size, avg_utility
    writer.writerow(["iteration", "cluster_size", "avg_utility"])
    for size in cluster_sizes:
        for i, it in enumerate(iterations):
            writer.writerow([it, size, float(avg_data[size][i])])

print(f"Averaged data saved to: {os.path.abspath(avg_csv_path)}")

# -----------------------------------------------
"""Figure: Average Utility per Iteration with Different Cluster Sizes"""
# -----------------------------------------------

# Create a new figure for the average performance plot
plt.figure(figsize=(6, 5))

# Plot the average lines for all cluster sizes
for size in cluster_sizes:
    color = colors.get(size, 'black')
    
    # Plot the average line
    plt.plot(iterations, avg_data[size], 
             label=f'$L_r = L_t = {size}$', 
             color=color)

# Increase font size for axis labels
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Average Utility', fontsize=14)

# Create a two-line title with different font sizes using matplotlib's text rendering
title_line1 = 'Utility vs Iteration for Different Cluster Sizes'

# Create an empty title with extra padding to make room for our custom title
plt.title("", pad=25)

# Add both title lines as text objects in the axes coordinates
ax = plt.gca()
ax.text(0.5, 1.02, title_line1, 
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes,
        fontsize=16)

# Increase font size for tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Increase legend font size and add a border
plt.legend(fontsize=12, frameon=True, edgecolor='black')
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation but move it to the y-label instead of at the top
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))  # Force scientific notation
plt.gca().yaxis.set_major_formatter(formatter)

# Hide the default offset text
ax.yaxis.offsetText.set_visible(False)

# Get the order of magnitude safely
max_value = max([max(avg_data[size]) for size in cluster_sizes]) if cluster_sizes else 0.0
if max_value > 0:
    power = int(np.floor(np.log10(max_value)))
else:
    power = 0

if power != 0:
    # Update y-label to include the scientific notation
    current_label = plt.gca().get_ylabel()
    plt.ylabel(f"{current_label} ($\\times 10^{{{power}}}$)", fontsize=14)

plt.tight_layout()
plt.savefig('average_utility_per_iteration.png', dpi=300)
plt.show()