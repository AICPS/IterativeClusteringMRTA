"""This function generates an assignment by creating random clusters of robots and tasks
Inputs:
        robot_list = list of robot objects
        task_list = list of task objects
        L_r = maximum number of robots in a cluster
        L_t = maximum number of tasks in a cluster
        kappa = number of different robot capabilities
        num_iterations = maximum number of clustering iterations to perform
        time_limit = maximum execution time in seconds (optional)
        printout = whether to print progress (optional)
        
The Algorithm is as follows:
1. Start with all robots and tasks in their own individual assignment grouping
2. For each iteration:
    1. Merge assignment groupings to create clusters
    2. Perform optimal assignment within each cluster
    3. Calculate and output the total reward of the current assignment
    4. Create assignment groupings based on the current assignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.IP_assignment import IP_assignment
from phase1.generate_clusters import generate_clusters_rand
import copy
import time

def cluster_assignment_rand(robot_list, task_list, hypes, max_iterations=10000, time_limit=5.0, printout=False):
    
    start_time = time.time()

    # Initialize the rewards, assignment vectors, and time tracking
    iteration_rewards = []
    iteration_assignments = []
    iteration_times = []  # List to track time per iteration

    """ 1. Start with all robots unassigned """
    assignment = {-1: [robot.get_id() for robot in robot_list]}
    for task in task_list:
        assignment[task.id] = []

    iteration = 0
    while time.time() - start_time < time_limit and iteration < max_iterations:

        if printout:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        """ 2. Generate clusters based on the current assignment """
        clusters = generate_clusters_rand(assignment, robot_list, task_list, hypes)

        """3. Perform optimal assignment within each cluster"""
        cluster_assignments = []
        cluster_assign_rewards = []
        for cluster in clusters:
            # L is max robots per task
            L = len(task_list[0].get_reward_matrix())-1
            
            assignment, reward = IP_assignment([robot_list[r] for r in cluster.robot_ids], [task_list[t] for t in cluster.task_ids], hypes)
            
            # Store cluster assignments and rewards
            cluster_assignments.append(assignment)
            cluster_assign_rewards.append(reward)
        
        """ Convert the cluster assignments to a single assignment """
        # Initialize the global assignment dictionary
        assignment = {-1: []}  # Start with empty list for unassigned robots
        for task in task_list:
            assignment[task.id] = []  # Empty list for each task

        for cluster_idx, cluster in enumerate(clusters):
            cluster_assignment = cluster_assignments[cluster_idx]
            
            # Add unassigned robots to the global assignment
            assignment[-1].extend(cluster_assignment.get(-1, []))
            
            # Add assigned robots to their respective tasks in the global assignment
            for task_id in cluster.task_ids:
                if task_id in cluster_assignment:
                    assignment[task_id].extend(cluster_assignment[task_id])

        # Output the results of the current iteration:
        total_reward = sum(cluster_assign_rewards)
        if printout:
            print(f"Total Reward: {total_reward}")
            print(f"Assignment: {assignment}")

        # Store the results of the current iteration
        iteration_rewards.append(total_reward)
        iteration_assignments.append(copy.deepcopy(assignment))
        iteration_times.append(time.time() - start_time)
    
        iteration += 1

    return total_reward, iteration_assignments, iteration_rewards, iteration_times