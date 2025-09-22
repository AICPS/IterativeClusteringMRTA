import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker
import csv
from algorithms.cluster_assignment_rand import cluster_assignment_rand
from algorithms.cluster_assignment_heuristic import cluster_assignment_heuristic
from algorithms.SA_efficient import SA_efficient
from algorithms.group_based_auction import group_based_auction
from algorithms.stochastic_greedy_search import stochastic_greedy_search
from algorithms.hedonic_game import hedonic_game
import test_utils as tu


"Test Parameters"
num_tests = 3
time_limit = 2  # seconds

"""Problem HyperParameters"""
nu = 50  # number of robots
mu = 25  # number of tasks
kappa = 3  # number of capabilities
max_x = 100
max_y = 100
max_d = np.sqrt(max_x**2 + max_y**2)  # maximum distance between any two points
L = 3  # maximum team size for a single task

"""Cluster Hyperparameters"""
#num_iterations = 100  # number of iterations to run
L_r = 6  # Max number of robots in a cluster
L_t = 3  # Max number of tasks in a cluster
temp = 20   # temperature ratio for softmax

"""SA Hyperparameters"""
initial_SA_temp = 10
SA_solutions_to_test = 80000

"""GA Hyperparameters"""
# For traditional:
population_size = 100
GA_solutions_to_test = 10000
mutation_rate = 0.8 
crossover_rate = 0.6

# For Replace Worst:
population_size = 200
GA_solutions_to_test = 10000
mutation_rate = 0.9 
crossover_rate = 0.6

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,
    'mu': mu,
    'kappa': kappa,
    'L': L,
    'L_r': L_r,
    'L_t': L_t,
    'max_d': max_d,
    'temp': temp,
    'initial_SA_temp': initial_SA_temp,
    'SA_solutions_to_test': SA_solutions_to_test,
    'population_size': population_size,
    'GA_solutions_to_test': GA_solutions_to_test,
    'mutation_rate': mutation_rate,
    'crossover_rate': crossover_rate,
    'max_x': max_x,
    'max_y': max_y
}

# Initialize a dictionary to store results for each method
# results = {
#     'heuristic_clustering': [],
#     'random_clustering': [],
#     #'spatial_cluster': [],
#     #'cap_match_cluster': [],
#     'simulated_annealing': [],
#     'group-based_auction': [],
#     #'stochastic_greedy': [],
#     'hedonic_game': []
#     #'GA': []
# }

results = {
    'HC': [],
    'RC': [],
    #'spatial_cluster': [],
    #'cap_match_cluster': [],
    'SA': [],
    'GBA': [],
    #'stochastic_greedy': [],
    'HG': []
    #'GA': []
}

# Generate a discrete task requirements problem instance
if kappa == 3:
    robot_list, task_list = tu.generate_problem_3_capabilities(hypes, max_x, max_y)
    # robot_list, task_list = tu.generate_random_problem_instance_3_mdr(hypes, max_x, max_y)
elif kappa == 2:
    robot_list, task_list = tu.generate_problem_2_capabilities(hypes, max_x, max_y)
else:
    print("Error: kappa must be 2 or 3")

for test in range(num_tests):
    print(f"Test: {test+1}")
    
    # Run each method
    for method in results.keys():
        if method == 'RC':
            _, _, rewards, times = cluster_assignment_rand(robot_list, task_list, hypes, time_limit=time_limit)
            print(f'Random Clustering Time: {times[-1]}')
            
        elif method == 'HC':
            _, _, rewards, times = cluster_assignment_heuristic(robot_list, task_list, hypes, time_limit=time_limit)
            print(f'Heuristic Clustering Time: {times[-1]}')
            
        elif method == 'SA':
            _, _, rewards, times = SA_efficient(robot_list, task_list, hypes, time_limit)
            print(f'SA Time: {times[-1]}')

        elif method == 'GBA':
            _, _, rewards, times = group_based_auction(robot_list, task_list, hypes)
            print(f'Auction Time: {times[-1]}')
            
        elif method == 'HG':
            _, _, rewards, times = hedonic_game(robot_list, task_list, hypes, time_limit=time_limit)
            print(f'Hedonic Game Time: {times[-1]}')

        results[method].append((rewards, times))
        print(f"{method.capitalize()} final reward: {rewards[-1]}")

# Define colors for each method
# colors = {
#     'random_clustering': 'black',
#     'heuristic_clustering': 'blue',
#     'spatial_cluster': 'green',
#     'cap_match_cluster': 'purple',
#     'simulated_annealing': 'red',
#     #'GA': 'brown',
#     'group-based_auction': 'green',
#     'hedonic_game': 'brown',
#     'stochastic_greedy': 'orange'
# }

colors = {
    'RC': 'black',
    'HC': 'blue',
    'SA': 'red',
    'GBA': 'green',
    'HG': 'brown',
}

# -----------------------------------------------
# First plot: Single test stair step
# -----------------------------------------------

# Create figure and axis objects explicitly for the first plot
fig1, ax1 = plt.subplots(figsize=(5, 4))

# Plot the first test results as stair steps
for method in results.keys():
    if results[method]:  # Check if there are results for this method
        rewards, times = results[method][0]  # Get first test results
        # Add a point at (0,0) to make the plot start from origin
        if times[0] > 0:
            plot_times = np.concatenate(([0], times))
            plot_rewards = np.concatenate(([0], rewards))
        else:
            plot_times = times
            plot_rewards = rewards
        
        ax1.step(plot_times, plot_rewards, where='post',
            label=method.replace('_', ' '),
            color=colors.get(method, 'black'))

# Formatting
ax1.set_xlabel('Time (seconds)', fontsize=14)
ax1.set_ylabel('Utility', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Set axis to start at 0
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# Use scientific notation for y-axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))  # Force scientific notation
ax1.yaxis.set_major_formatter(formatter)

# Adjust the position of the offset text
ax1.yaxis.offsetText.set_fontsize(10)  # Make the exponent text slightly smaller
ax1.yaxis.offsetText.set_position((0, 1))  # Move it to the top

# Add legend with increased font size and border
ax1.legend(fontsize=12, frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig('first_test_comparison.png', dpi=300)
plt.show()


# -----------------------------------------------
# Second plot: Average Performance
# -----------------------------------------------

# Create figure and axis objects explicitly for the second plot
fig2, ax2 = plt.subplots(figsize=(5, 4))

# Define a common time grid for interpolation
max_time = time_limit
time_grid = np.linspace(0, max_time, 1000)

# Dictionary to store average data for each method
avg_data = {method: np.zeros(len(time_grid)) for method in results.keys() if results[method]}

# Collect all interpolated rewards and calculate averages
for method in results.keys():
    if not results[method]:
        continue
        
    # Initialize array to store interpolated rewards
    all_interpolated_rewards = []
    
    # Process each test
    for test_idx in range(len(results[method])):
        rewards, times = results[method][test_idx]
        
        # Create a step function for interpolation
        def step_function(t):
            idx = np.searchsorted(times, t, side='right') - 1
            if idx < 0:
                return 0  # Return 0 for times before the first recorded time
            return rewards[idx]
        
        # Apply the step function to each time in the grid
        interpolated_rewards = [step_function(t) for t in time_grid]
        all_interpolated_rewards.append(interpolated_rewards)
    
    # Calculate the average at each time point
    if all_interpolated_rewards:
        all_rewards_array = np.array(all_interpolated_rewards)
        avg_data[method] = np.mean(all_rewards_array, axis=0)

# Plot the average lines for all methods
for method in results.keys():
    if method in avg_data:
        color = colors.get(method, 'black')
        
        # Plot the average line
        ax2.plot(time_grid, avg_data[method],
            label=method.replace('_', ' '),
            color=color)

# Formatting
ax2.set_xlabel('Time (seconds)', fontsize=14)
ax2.set_ylabel('Average Utility', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation for y-axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))  # Force scientific notation
ax2.yaxis.set_major_formatter(formatter)

# Adjust the position of the offset text
ax2.yaxis.offsetText.set_fontsize(10)  # Make the exponent text slightly smaller
ax2.yaxis.offsetText.set_position((0, 1))  # Move it to the top

# Add legend with increased font size and border
# ax2.legend(fontsize=11, frameon=True, edgecolor='black')

ax2.legend(loc='center right', fontsize=11, frameon=True, edgecolor='black')

# ax2.legend(
#     fontsize=10,
#     frameon=True,
#     edgecolor='black',
#     loc='center left',          # anchor legend box by its left edge
#     bbox_to_anchor=(1.02, 0.5)  # place it just outside the right edge, vertically centered
# )

plt.tight_layout()
plt.savefig('average_performance.png', dpi=300)
plt.show()






# # -----------------------------------------------
# # Second plot: Average Performance
# # -----------------------------------------------

# # Create figure and axis objects explicitly for the second plot
# fig2, ax2 = plt.subplots(figsize=(5, 4))

# # Define a common time grid for interpolation
# max_time = time_limit
# time_grid = np.linspace(0, max_time, 1000)

# # Dictionary to store average data for each method
# avg_data = {method: np.zeros(len(time_grid)) for method in results.keys() if results[method]}

# # Collect all interpolated rewards and calculate averages
# for method in results.keys():
#     if not results[method]:
#         continue
        
#     # Initialize array to store interpolated rewards
#     all_interpolated_rewards = []
    
#     # Process each test
#     for test_idx in range(len(results[method])):
#         rewards, times = results[method][test_idx]
        
#         # Create a step function for interpolation
#         def step_function(t):
#             idx = np.searchsorted(times, t, side='right') - 1
#             if idx < 0:
#                 return 0  # Return 0 for times before the first recorded time
#             return rewards[idx]
        
#         # Apply the step function to each time in the grid
#         interpolated_rewards = [step_function(t) for t in time_grid]
#         all_interpolated_rewards.append(interpolated_rewards)
    
#     # Calculate the average at each time point
#     if all_interpolated_rewards:
#         all_rewards_array = np.array(all_interpolated_rewards)
#         avg_data[method] = np.mean(all_rewards_array, axis=0)

# # Plot the average lines for all methods
# for method in results.keys():
#     if method in avg_data:
#         color = colors.get(method, 'black')
        
#         # Plot the average line
#         ax2.plot(time_grid, avg_data[method], 
#                 label=method.replace('_', ' ').title(), 
#                 color=color)

# # Formatting
# ax2.set_xlabel('Time (seconds)', fontsize=14)
# ax2.set_ylabel('Average Utility', fontsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=12)
# ax2.grid(True, linestyle='--', alpha=0.7)

# # Use scientific notation for y-axis
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1,1))  # Force scientific notation
# ax2.yaxis.set_major_formatter(formatter)

# # Adjust the position of the offset text
# ax2.yaxis.offsetText.set_fontsize(10)

# # Add legend STACKED VERTICALLY on the right side
# ax2.legend(
#     fontsize=10,
#     frameon=True,
#     edgecolor='black',
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5)
# )

# plt.tight_layout()
# plt.savefig('average_performance.png', dpi=300, bbox_inches="tight")
# plt.show()








# -----------------------------------------------
# Print statistics at specific time stamps
# -----------------------------------------------
print("\n" + "="*70)
print("Utility statistics at specific time stamps:")
print("="*70)

# Define the time stamps we want to analyze
time_stamps = [0.1, 0.25, 0.5, 1.0, 2.0]  # 1 second and 2 seconds

for time_stamp in time_stamps:
    print(f"\nAt time = {time_stamp} seconds:")
    print("-" * 60)
    
    for method in results.keys():
        if not results[method]:
            continue
            
        # Collect utilities at this time stamp across all tests
        utilities_at_timestamp = []
        
        for test_idx in range(len(results[method])):
            rewards, times = results[method][test_idx]
            
            # Find the utility at or just before this time stamp
            idx = np.searchsorted(times, time_stamp, side='right') - 1
            if idx >= 0:  # Make sure we found a valid index
                utilities_at_timestamp.append(rewards[idx])
            else:
                utilities_at_timestamp.append(0)  # No reward recorded before this time
        
        # Calculate statistics
        if utilities_at_timestamp:
            avg_utility = np.mean(utilities_at_timestamp)
            max_utility = np.max(utilities_at_timestamp)
            std_utility = np.std(utilities_at_timestamp)
            
            print(f"{method.replace('_', ' ').title()}:")
            print(f"  Average: {avg_utility:.2f}")
            print(f"  Maximum: {max_utility:.2f}")
            print(f"  Std Dev: {std_utility:.2f}")



# Create filenames with the requested naming format
single_test_filename = f"single_test_{kappa}_{nu}_{mu}.csv"
avg_rewards_filename = f"avg_rewards_{kappa}_{nu}_{mu}_{num_tests}.csv"

# Export single test data
with open(single_test_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header with method names
    header = ['time']
    for method in results.keys():
        if results[method]:
            header.append(method)
    writer.writerow(header)
    
    # Create a dictionary to store all time points
    all_times = set()
    method_data = {}
    
    # Collect all time points and prepare data
    for method in results.keys():
        if results[method]:
            rewards, times = results[method][0]  # Get first test results
            method_data[method] = {t: r for t, r in zip(times, rewards)}
            all_times.update(times)
    
    # Sort all time points
    all_times = sorted(all_times)
    
    # Write data rows
    for t in all_times:
        row = [t]
        for method in results.keys():
            if method in method_data and t in method_data[method]:
                row.append(method_data[method][t])
            elif method in method_data:
                # Find the most recent reward before this time
                prev_times = [prev_t for prev_t in method_data[method].keys() if prev_t <= t]
                if prev_times:
                    most_recent_t = max(prev_times)
                    row.append(method_data[method][most_recent_t])
                else:
                    row.append(0)  # No previous reward
            else:
                row.append('')  # Method has no data
        writer.writerow(row)
    
    print(f"Single test data saved to {single_test_filename}")

# Export averaged interpolated data
with open(avg_rewards_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header with method names
    header = ['time']
    for method in results.keys():
        if results[method]:
            header.append(method)
    writer.writerow(header)
    
    # Prepare the averaged data
    avg_data = {}
    for method in results.keys():
        if not results[method]:
            continue
            
        # Initialize array to store interpolated rewards
        all_interpolated_rewards = []
        
        for test_idx in range(len(results[method])):
            rewards, times = results[method][test_idx]
            
            # Create a step function for interpolation
            def step_function(t):
                idx = np.searchsorted(times, t, side='right') - 1
                if idx < 0:
                    return 0  # Return 0 for times before the first recorded time
                return rewards[idx]
            
            # Apply the step function to each time in the grid
            interpolated_rewards = [step_function(t) for t in time_grid]
            all_interpolated_rewards.append(interpolated_rewards)
        
        # Calculate the average reward at each time point
        if all_interpolated_rewards:
            avg_rewards = np.mean(all_interpolated_rewards, axis=0)
            avg_data[method] = avg_rewards
    
    # Write data rows
    for i, t in enumerate(time_grid):
        row = [t]
        for method in results.keys():
            if method in avg_data:
                row.append(avg_data[method][i])
            else:
                row.append('')  # Method has no data
        writer.writerow(row)
    
    print(f"Average rewards data saved to {avg_rewards_filename}")