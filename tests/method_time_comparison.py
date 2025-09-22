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
from algorithms.hedonic_game import hedonic_game
import test_utils as tu


"Test Parameters"
num_tests = 100
time_limit = 2  # seconds for each method

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
    'max_x': max_x,
    'max_y': max_y
}

# Store results for each method
results = {
    'HC': [],
    'RC': [],
    'SA': [],
    'GBA': [],
    'HG': []
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

colors = {
    'RC': 'black',
    'HC': 'blue',
    'SA': 'red',
    'GBA': 'green',
    'HG': 'brown',
}


# -----------------------------------------------
# Plot: Average Performance
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

ax2.legend(loc='center right', fontsize=11, frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig('average_performance.png', dpi=300)
plt.show()

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


# -----------------------------------------------
# Export results to CSV files
# -----------------------------------------------

# Create filenames with the requested naming format
avg_rewards_filename = f"avg_rewards_{kappa}_{nu}_{mu}_{num_tests}.csv"

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