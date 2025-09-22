import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import csv
import pickle
from algorithms.cluster_assignment_rand import cluster_assignment_rand
from algorithms.cluster_assignment_heuristic import cluster_assignment_heuristic
from algorithms.SA_efficient import SA_efficient
import test_utils as tu

# Test Parameters
num_tests = 100
time_limit = 10  # seconds

# Problem sizes
problem_sizes = [
    (250, 125),
    (500, 250),
    (750, 375),
    (1000, 500),
]

# Specify which methods to run and plot
methods_to_use = [
    'random_clustering',
    'heuristic_clustering',
#    'simulated_annealing'
]

# Common Problem HyperParameters
kappa = 3  # number of capabilities
max_x = 100
max_y = 100
max_d = np.sqrt(max_x**2 + max_y**2)  # maximum distance between any two points
L = 3  # maximum team size for a single task

# Cluster Hyperparameters
L_r = 6  # Max number of robots in a cluster
L_t = 3  # Max number of tasks in a cluster
temp = 20   # temperature ratio for softmax

# SA Hyperparameters
initial_SA_temp = 10
SA_solutions_to_test = 80000

# Colors for each method
colors = {
    'random_clustering': 'black',
    'heuristic_clustering': 'blue',
    'simulated_annealing': 'red'
}

# Line styles for each problem size
line_styles = [':', '--', '-.', '-']

# Initialize results dictionary
results = {size: {method: [] for method in methods_to_use} for size in problem_sizes}

# Run tests for each problem size
for size in problem_sizes:
    nu, mu = size
    print(f"\nRunning tests for {nu} robots and {mu} tasks")
    
    # Update hyperparameters
    hypes = {
        'nu': nu, 'mu': mu, 'kappa': kappa, 'L': L, 'L_r': L_r, 'L_t': L_t,
        'max_d': max_d, 'temp': temp, 'initial_SA_temp': initial_SA_temp,
        'SA_solutions_to_test': SA_solutions_to_test, 'max_x': max_x, 'max_y': max_y
    }
    
    # Generate problem instance
    robot_list, task_list = tu.generate_problem_3_capabilities(hypes, max_x, max_y)

    for test in range(num_tests):
        print(f"Test: {test+1}")
        
        # Run each method
        for method in methods_to_use:
            if method == 'random_clustering':
                _, _, rewards, times = cluster_assignment_rand(robot_list, task_list, hypes, time_limit=time_limit)
            elif method == 'heuristic_clustering':
                _, _, rewards, times = cluster_assignment_heuristic(robot_list, task_list, hypes, time_limit=time_limit)
            elif method == 'simulated_annealing':
                _, _, rewards, times = SA_efficient(robot_list, task_list, hypes, time_limit)
            
            results[size][method].append((rewards, times))
            print(f"{method.capitalize()} final reward: {rewards[-1]}")

# Define a common time grid for interpolation
time_grid = np.linspace(0, time_limit, 1000)

# Prepare data for line plot
line_plot_data = {
    'time_limit': time_limit,
    'num_tests': num_tests,
    'problem_sizes': problem_sizes,
    'methods_to_use': methods_to_use,
    'colors': colors,
    'line_styles': line_styles,
    'time_grid': time_grid,
    'avg_rewards_by_size_method': {}
}

# Calculate average rewards for line plot
for size in problem_sizes:
    line_plot_data['avg_rewards_by_size_method'][size] = {}
    for method in methods_to_use:
        # Calculate average rewards using stairstep approach
        avg_rewards = np.zeros(len(time_grid))
        for test_rewards, test_times in results[size][method]:
            current_reward = 0
            for j, t in enumerate(time_grid):
                idx = np.searchsorted(test_times, t, side='right') - 1
                if idx >= 0:
                    current_reward = test_rewards[idx]
                avg_rewards[j] += current_reward
        avg_rewards /= num_tests
        line_plot_data['avg_rewards_by_size_method'][size][method] = avg_rewards

# Save line plot data
line_plot_filename = f"scalability_linePlot_{time_limit}_{num_tests}.pkl"
with open(line_plot_filename, 'wb') as f:
    pickle.dump(line_plot_data, f)

# Prepare data for contour plot
contour_time_grid = np.linspace(0, time_limit, 100)
problem_sizes_values = np.array([size[0] for size in problem_sizes])
rewards_matrix = np.zeros((len(problem_sizes), len(contour_time_grid)))

# Fill the rewards matrix
for i, size in enumerate(problem_sizes):
    # Calculate average rewards using stairstep approach
    avg_rewards = np.zeros(len(contour_time_grid))
    for test_rewards, test_times in results[size]['random_clustering']:
        current_reward = 0
        for j, t in enumerate(contour_time_grid):
            idx = np.searchsorted(test_times, t, side='right') - 1
            if idx >= 0:
                current_reward = test_rewards[idx]
            avg_rewards[j] += current_reward
    avg_rewards /= num_tests
    rewards_matrix[i, :] = avg_rewards

contour_data = {
    'time_limit': time_limit,
    'num_tests': num_tests,
    'problem_sizes': problem_sizes,
    'contour_time_grid': contour_time_grid,
    'problem_sizes_values': problem_sizes_values,
    'rewards_matrix': rewards_matrix
}

# Save contour plot data
contour_filename = f"scalability_contour_{time_limit}_{num_tests}.pkl"
with open(contour_filename, 'wb') as f:
    pickle.dump(contour_data, f)

print(f"\nData saved to {line_plot_filename} and {contour_filename}")

# Now create and display the plots as before
# (rest of the plotting code remains the same)

# Create figure and axis objects explicitly
fig, ax = plt.subplots(figsize=(8, 5))

# Define a common time grid for interpolation
time_grid = np.linspace(0, time_limit, 1000)

# Plot average results for each problem size and method
for i, size in enumerate(problem_sizes):
    nu, mu = size
    line_style = line_styles[i]
    
    for method in methods_to_use:
        color = colors[method]
        
        # Calculate average rewards using stairstep approach
        avg_rewards = np.zeros(len(time_grid))
        for test_rewards, test_times in results[size][method]:
            current_reward = 0
            for j, t in enumerate(time_grid):
                idx = np.searchsorted(test_times, t, side='right') - 1
                if idx >= 0:
                    current_reward = test_rewards[idx]
                avg_rewards[j] += current_reward
        avg_rewards /= num_tests
        
        # Plot the average line
        ax.plot(time_grid, avg_rewards, color=color, linestyle=line_style)

# Formatting
ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Average Utility', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation for y-axis
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax.yaxis.set_major_formatter(formatter)

# Adjust the position of the offset text
ax.yaxis.offsetText.set_fontsize(10)
ax.yaxis.offsetText.set_position((0, 1))

# Create line style legend
line_legend = [
    Line2D([0], [0], color='black', linestyle=':', lw=2, label='250 robots, 125 tasks'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='500 robots, 250 tasks'),
    Line2D([0], [0], color='black', linestyle='-.', lw=2, label='750 robots, 375 tasks'),
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='1000 robots, 500 tasks')
]

# Only create and display methods legend if there's more than one method
if len(methods_to_use) > 1:
    # Create color legend for methods
    color_legend = [
        Line2D([0], [0], color=colors[method], lw=2, label=method.replace('_', ' ').title())
        for method in methods_to_use
    ]
    
    # Place color legend in bottom right (left side)
    first_legend = ax.legend(handles=color_legend, 
                             loc='lower right', 
                             bbox_to_anchor=(0.65, 0.02),
                             fontsize=8, frameon=True, edgecolor='black',
                             title="Methods")
    
    # Add the first legend manually to the axis
    ax.add_artist(first_legend)
    
    # Place line style legend to the right of the color legend
    second_legend = ax.legend(handles=line_legend,
                              loc='lower right', 
                              bbox_to_anchor=(0.98, 0.02),
                              fontsize=8, frameon=True, edgecolor='black',
                              title="Problem Sizes")
else:
    # If only one method, just show the problem sizes legend
    ax.legend(handles=line_legend,
              loc='lower right',
              fontsize=8, frameon=True, edgecolor='black',
              title="Problem Sizes")

plt.tight_layout()
plt.savefig('comparison_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# Fig 2 Contours
# -------------------------------------------------------------

# Create a second figure for the contour plot
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Define a finer time grid for the contour plot
contour_time_grid = np.linspace(0, time_limit, 100)

# Create arrays to hold the data for the contour plot
problem_sizes_values = np.array([size[0] for size in problem_sizes])  # Number of robots
rewards_matrix = np.zeros((len(problem_sizes), len(contour_time_grid)))

# Fill the rewards matrix
for i, size in enumerate(problem_sizes):
    # Calculate average rewards using stairstep approach
    avg_rewards = np.zeros(len(contour_time_grid))
    for test_rewards, test_times in results[size]['random_clustering']:
        current_reward = 0
        for j, t in enumerate(contour_time_grid):
            idx = np.searchsorted(test_times, t, side='right') - 1
            if idx >= 0:
                current_reward = test_rewards[idx]
            avg_rewards[j] += current_reward
    avg_rewards /= num_tests
    rewards_matrix[i, :] = avg_rewards

# Create a meshgrid for the contour plot
X, Y = np.meshgrid(contour_time_grid, problem_sizes_values)

# Create the contour plot
contour = ax2.contourf(X, Y, rewards_matrix, 20, cmap='viridis')

# Add a colorbar
cbar = fig2.colorbar(contour, ax=ax2)
cbar.set_label('Average Utility', fontsize=12)

# Formatting
ax2.set_xlabel('Time (seconds)', fontsize=14)
ax2.set_ylabel('Problem Size: (Robots, Tasks)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
#ax2.set_title('Random Clustering Performance vs Time and Problem Size', fontsize=14)

# Add problem size labels on y-axis
ax2.set_yticks(problem_sizes_values)
ax2.set_yticklabels([f"({nu}, {mu})" for nu, mu in problem_sizes])

plt.tight_layout()
plt.savefig('contour_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics at specific time stamps
print("\n" + "="*70)
print("Utility statistics at specific time stamps:")
print("="*70)

time_stamps = [0.1, 0.25, 0.5, 1.0]

for size in problem_sizes:
    nu, mu = size
    print(f"\nProblem size: {nu} robots, {mu} tasks")
    print("="*50)
    
    for time_stamp in time_stamps:
        if time_stamp <= time_limit:  # Only print for timestamps within the time limit
            print(f"\nAt time = {time_stamp} seconds:")
            print("-" * 40)
            
            for method in methods_to_use:
                utilities_at_timestamp = []
                
                for rewards, times in results[size][method]:
                    idx = np.searchsorted(times, time_stamp, side='right') - 1
                    if idx >= 0:
                        utilities_at_timestamp.append(rewards[idx])
                    else:
                        utilities_at_timestamp.append(0)
                
                avg_utility = np.mean(utilities_at_timestamp)
                max_utility = np.max(utilities_at_timestamp)
                std_utility = np.std(utilities_at_timestamp)
                
                print(f"{method.replace('_', ' ').title()}:")
                print(f"  Average: {avg_utility:.2f}")
                print(f"  Maximum: {max_utility:.2f}")
                print(f"  Std Dev: {std_utility:.2f}")