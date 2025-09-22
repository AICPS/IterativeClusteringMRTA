"""
This file iteratively clusters and reassigns robots to tasks and compares with optimal assignment
"""

import sys
import os
import time
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase2.IP_assignment import IP_assignment
from algorithms.cluster_assignment_rand import cluster_assignment_rand
from algorithms.cluster_assignment_heuristic import cluster_assignment_heuristic
import test_utils as tu

"""HyperParameters"""
nu = 10  # number of robots # was 10
mu = 5  # number of tasks  # was 5
kappa = 3  # number of capabilities
L = 3  # maximum team size for a single task
L_r = 6  # Max number of robots in a cluster
L_t = 3  # Max number of tasks in a cluster
time_limit = 2  # maximum execution time in seconds
num_tests = 2  # number of random tests to run
temp = 20   # temperature ratio for softmax

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
    'L_t': L_t,   # maximum number of tasks in a cluster
    'L_r': L_r,   # maximum number of robots in a cluster
    'temp': temp,  # temperature for heuristic method
}

# Define the environment size
max_x = 100
max_y = 100

# Run multiple tests and collect statistics
random_matches = 0
heuristic_matches = 0
total_tests = 0
optimal_rewards = []
random_cluster_rewards = []
heuristic_cluster_rewards = []

# Track execution times
random_times = []
heuristic_times = []
optimal_times = []

# Prepare CSV file
csv_filename = "iteration_optimal_compared.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow([
        'Test', 'Optimal_Reward', 'Random_Reward', 'Heuristic_Reward', 
        'Random_Match', 'Heuristic_Match', 'Optimal_Time', 'Random_Time', 'Heuristic_Time'
    ])

    for test in range(num_tests):
        print(f"\n--- Test {test + 1}/{num_tests} ---")
        
        # Generate the problem instancs
        if kappa == 2:
            robot_list, task_list = tu.generate_random_problem_instance_2(hypes, max_x, max_y)
        elif kappa == 3:
            robot_list, task_list = tu.generate_random_problem_instance_3(hypes, max_x, max_y)
        else:
            print("Error: kappa must be 2 or 3")
            exit(1)
        
        # Run random clustering method
        random_start_time = time.time()
        random_total_reward, random_iteration_assignments, random_iteration_rewards, random_iteration_times = cluster_assignment_rand(
            robot_list, task_list, hypes, time_limit=time_limit
        )
        random_total_time = time.time() - random_start_time
        random_times.append(random_total_time)
        
        # Run heuristic clustering method
        heuristic_start_time = time.time()
        heuristic_total_reward, heuristic_iteration_assignments, heuristic_iteration_rewards, heuristic_iteration_times = cluster_assignment_heuristic(
            robot_list, task_list, hypes, time_limit=time_limit
        )
        heuristic_total_time = time.time() - heuristic_start_time
        heuristic_times.append(heuristic_total_time)
        
        # Get the final rewards
        random_final_reward = random_iteration_rewards[-1]
        heuristic_final_reward = heuristic_iteration_rewards[-1]
        
        # Calculate optimal assignment
        optimal_start_time = time.time()
        optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, hypes)
        optimal_total_time = time.time() - optimal_start_time
        optimal_times.append(optimal_total_time)

        # Compare results
        print(f"Optimal Reward: {optimal_reward}")
        print(f"Random Clustering Final Reward: {random_final_reward}")
        print(f"Heuristic Clustering Final Reward: {heuristic_final_reward}")
        
        # Report execution times
        print(f"Execution Times - Random: {random_total_time:.4f}s, Heuristic: {heuristic_total_time:.4f}s, Optimal: {optimal_total_time:.4f}s")
        
        # Check if the methods match the optimal reward (using a small epsilon for floating point comparison)
        epsilon = 1e-6
        
        random_match = abs(random_final_reward - optimal_reward) < epsilon
        if random_match:
            random_matches += 1
            print("Random Clustering: MATCH!")
        else:
            print(f"Random Clustering Difference: {optimal_reward - random_final_reward}")
        
        heuristic_match = abs(heuristic_final_reward - optimal_reward) < epsilon
        if heuristic_match:
            heuristic_matches += 1
            print("Heuristic Clustering: MATCH!")
        else:
            print(f"Heuristic Clustering Difference: {optimal_reward - heuristic_final_reward}")
        
        total_tests += 1
        optimal_rewards.append(optimal_reward)
        random_cluster_rewards.append(random_final_reward)
        heuristic_cluster_rewards.append(heuristic_final_reward)
        
        # Write results to CSV
        csv_writer.writerow([
            test + 1, 
            optimal_reward, 
            random_final_reward, 
            heuristic_final_reward,
            int(random_match),
            int(heuristic_match),
            optimal_total_time,
            random_total_time,
            heuristic_total_time
        ])

# Calculate statistics
random_match_percentage = (random_matches / total_tests) * 100
heuristic_match_percentage = (heuristic_matches / total_tests) * 100

avg_optimal_reward = np.mean(optimal_rewards)
avg_random_reward = np.mean(random_cluster_rewards)
avg_heuristic_reward = np.mean(heuristic_cluster_rewards)

# Calculate utility ratio by summing all utilities and then dividing
sum_optimal_rewards = np.sum(optimal_rewards)
sum_random_rewards = np.sum(random_cluster_rewards)
sum_heuristic_rewards = np.sum(heuristic_cluster_rewards)

# Avoid division by zero
if sum_optimal_rewards > 0:
    random_performance_ratio = (sum_random_rewards / sum_optimal_rewards) * 100
    heuristic_performance_ratio = (sum_heuristic_rewards / sum_optimal_rewards) * 100
else:
    random_performance_ratio = 100
    heuristic_performance_ratio = 100

# Calculate average execution times
avg_random_time = np.mean(random_times)
avg_heuristic_time = np.mean(heuristic_times)
avg_optimal_time = np.mean(optimal_times)

print("\n--- Final Statistics ---")
print(f"Total Tests: {total_tests}")

print("\nRandom Clustering:")
print(f"  Matches with Optimal: {random_matches}")
print(f"  Match Percentage: {random_match_percentage:.2f}%")
print(f"  Average Reward: {avg_random_reward:.2f}")
print(f"  Performance Ratio: {random_performance_ratio:.2f}% of optimal")
print(f"  Average Execution Time: {avg_random_time:.4f} seconds")

print("\nHeuristic Clustering:")
print(f"  Matches with Optimal: {heuristic_matches}")
print(f"  Match Percentage: {heuristic_match_percentage:.2f}%")
print(f"  Average Reward: {avg_heuristic_reward:.2f}")
print(f"  Performance Ratio: {heuristic_performance_ratio:.2f}% of optimal")
print(f"  Average Execution Time: {avg_heuristic_time:.4f} seconds")

print("\nOptimal Assignment:")
print(f"  Average Reward: {avg_optimal_reward:.2f}")
print(f"  Average Execution Time: {avg_optimal_time:.4f} seconds")

# Additional statistics
print("\n--- Comparison Statistics ---")
if avg_heuristic_reward > avg_random_reward:
    improvement = ((avg_heuristic_reward - avg_random_reward) / avg_random_reward) * 100
    print(f"Heuristic outperforms Random by {improvement:.2f}%")
elif avg_random_reward > avg_heuristic_reward:
    improvement = ((avg_random_reward - avg_heuristic_reward) / avg_heuristic_reward) * 100
    print(f"Random outperforms Heuristic by {improvement:.2f}%")
else:
    print("Random and Heuristic perform equally on average")

# Calculate how many times each method was better
heuristic_wins = sum(1 for h, r in zip(heuristic_cluster_rewards, random_cluster_rewards) if h > r)
random_wins = sum(1 for h, r in zip(heuristic_cluster_rewards, random_cluster_rewards) if r > h)
ties = sum(1 for h, r in zip(heuristic_cluster_rewards, random_cluster_rewards) if abs(h - r) < epsilon)

print(f"Heuristic better than Random: {heuristic_wins} times ({heuristic_wins/total_tests*100:.2f}%)")
print(f"Random better than Heuristic: {random_wins} times ({random_wins/total_tests*100:.2f}%)")
print(f"Equal performance: {ties} times ({ties/total_tests*100:.2f}%)")

# Time comparison
print("\n--- Execution Time Comparison ---")
print(f"Random vs Optimal: Random is {avg_optimal_time/avg_random_time:.2f}x faster")
print(f"Heuristic vs Optimal: Heuristic is {avg_optimal_time/avg_heuristic_time:.2f}x faster")
if avg_heuristic_time > avg_random_time:
    print(f"Random vs Heuristic: Random is {avg_heuristic_time/avg_random_time:.2f}x faster")
else:
    print(f"Heuristic vs Random: Heuristic is {avg_random_time/avg_heuristic_time:.2f}x faster")

# Save summary statistics to CSV
with open("iteration_summary_stats.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Metric', 'Random', 'Heuristic', 'Optimal'])
    csv_writer.writerow(['Match Percentage', f"{random_match_percentage:.2f}%", f"{heuristic_match_percentage:.2f}%", "100%"])
    csv_writer.writerow(['Average Reward', f"{avg_random_reward:.4f}", f"{avg_heuristic_reward:.4f}", f"{avg_optimal_reward:.4f}"])
    csv_writer.writerow(['Performance Ratio', f"{random_performance_ratio:.2f}%", f"{heuristic_performance_ratio:.2f}%", "100%"])
    csv_writer.writerow(['Average Time (s)', f"{avg_random_time:.4f}", f"{avg_heuristic_time:.4f}", f"{avg_optimal_time:.4f}"])
    csv_writer.writerow(['Sum of Rewards', f"{sum_random_rewards:.4f}", f"{sum_heuristic_rewards:.4f}", f"{sum_optimal_rewards:.4f}"])

print(f"\nResults saved to {csv_filename} and iteration_summary_stats.csv")