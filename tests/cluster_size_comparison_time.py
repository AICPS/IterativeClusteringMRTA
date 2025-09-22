"""
This file compares the results of the random merging method with different group size limits,
and saves the averaged time-series data required to recreate the figure to a CSV file:
cluster_size_comparison_time_{tests}_{nu}_{mu}.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot
import matplotlib.pyplot as plt
import pickle
from algorithms.cluster_assignment_rand import cluster_assignment_rand
import test_utils as tu
from matplotlib import ticker
import csv  # NEW: for CSV writing

"""HyperParameters"""
nu = 100 #number of robots
mu = 50 # number of tasks
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
time_limit = 1

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

        # Perform random iterative assignment (time-limited)
        total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
            robot_list, task_list, hypes, time_limit=time_limit
        )
        
        # Store the iteration rewards and times for this test and cluster size
        results[cluster_size].append(iteration_rewards)
        time_results[cluster_size].append((iteration_rewards, iteration_times))

# -----------------------------------------------
"""Figure: Average Utility vs Time with Different Cluster Sizes"""
# -----------------------------------------------

# Create a new figure for the average performance plot
plt.figure(figsize=(6, 5))

# Define a common time grid for interpolation
max_time = time_limit
time_grid = np.linspace(0, max_time, 1000)

# Dictionary to store average data for each cluster size
avg_time_data = {size: np.zeros(len(time_grid)) for size in cluster_sizes}

# Process time-based data for each cluster size
for size in cluster_sizes:
    all_interpolated_rewards = []
    
    # Process each test
    for test_idx in range(len(time_results[size])):
        rewards, times = time_results[size][test_idx]
        times = np.asarray(times, dtype=float)
        rewards = np.asarray(rewards, dtype=float)

        # Create a step function for interpolation
        def step_function(t):
            idx = np.searchsorted(times, t, side='right') - 1
            if idx < 0:
                return 0.0  # Return 0 for times before the first recorded time
            return rewards[idx]
        
        # Apply the step function to each time in the grid
        interpolated_rewards = [step_function(t) for t in time_grid]
        all_interpolated_rewards.append(interpolated_rewards)
    
    # Calculate the average at each time point
    if all_interpolated_rewards:
        all_rewards_array = np.array(all_interpolated_rewards, dtype=float)
        avg_time_data[size] = np.mean(all_rewards_array, axis=0)

# -----------------------------------------------
# NEW: Save averaged time-series data to CSV (long-form) for figure reproduction
# -----------------------------------------------
csv_filename = f"cluster_size_comparison_time_{num_tests}_{nu}_{mu}.csv"
with open(csv_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Long-form header: time, cluster_size, avg_utility
    writer.writerow(["time", "cluster_size", "avg_utility"])
    for size in cluster_sizes:
        for i, t in enumerate(time_grid):
            writer.writerow([float(t), size, float(avg_time_data[size][i])])

print(f"Averaged time-series data saved to: {os.path.abspath(csv_filename)}")

# Plot the average lines for all cluster sizes
for size in cluster_sizes:
    color = colors.get(size, 'black')
    
    # Plot the average line
    plt.plot(time_grid, avg_time_data[size], 
             label=f'$L_r = L_t = {size}$', 
             color=color)

# Increase font size for axis labels
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Average Utility', fontsize=14)

# Create a title with different font sizes using matplotlib's text rendering
title_line1 = 'Utility vs Time for Different Cluster Sizes'

# Create an empty title with extra padding to make room for our custom title
plt.title("", pad=25)

# Add title as text object in the axes coordinates
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

# Get the order of magnitude
max_value = max([max(avg_time_data[size]) for size in cluster_sizes]) if cluster_sizes else 0.0
power = int(np.floor(np.log10(max_value))) if max_value > 0 else 0
if power != 0:
    # Update y-label to include the scientific notation
    current_label = plt.gca().get_ylabel()
    plt.ylabel(f"{current_label} ($\\times 10^{{{power}}}$)", fontsize=14)

plt.tight_layout()
plt.savefig('average_utility_vs_time.png', dpi=300)
plt.show()

# Print final utility values for each cluster size
print("\nFinal average utility for each cluster size:")
for size in cluster_sizes:
    final_utility = avg_time_data[size][-1]
    print(f"Cluster Size {size}: {final_utility:.2f}")