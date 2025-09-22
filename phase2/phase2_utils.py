from itertools import combinations
import numpy as np
import math

"""This code generates integer partitions of n into m parts, where the first part is unlimited in size."""
def generate_partitions(n, m, L):
    def partition_helper(n_remaining, m_remaining, current_partition, partitions):
        if m_remaining == 0:
            if n_remaining == 0:
                partitions.append(current_partition[:])
            return
        
        else:  # Other groups are limited by L
            for i in range(0, min(n_remaining, L) + 1):
                current_partition.append(i)
                partition_helper(n_remaining - i, m_remaining - 1, current_partition, partitions)
                current_partition.pop()

    partitions = []
    for i in range(0,n+1):
        partition_helper(n-i, m, [i], partitions)
    return partitions

"""This function calculates the net reward for a given robot team and task."""
def calculate_net_reward(robot_team, task):
    if len(robot_team) < 1:
        return 0
    else:
        # Calculate team composition vector
        team_capabilities = np.zeros(len(robot_team[0].get_capabilities()), dtype=np.int32)
        for robot_idx in range(len(robot_team)):
            team_capabilities += robot_team[robot_idx].get_capabilities()
        capability_value =  task.get_reward(*team_capabilities)
    
        # Calculate cost assuming cost = distance traveled                
        cost = 0
        for robot_idx in range(len(robot_team)):
            cost += math.dist(robot_team[robot_idx].get_location(), task.get_location())

        net_reward = capability_value - cost
        # net_reward = capability_value
        return net_reward 

""" Search a given integer partition for the best assignment.
    Output assignment as a dictionary: Keys are task IDs, values are lists of robot IDs assigned to that task."""
def partition_search(robots, tasks, partition):
    # Initialize the assignment dictionary
    assignment = {-1: []}  # Start with empty list for unassigned robots
    for task in tasks:
        assignment[task.get_id()] = []  # Empty list for each task

    # Base case: there are no tasks
    if len(tasks) == 0:
        assignment[-1] = [robot.get_id() for robot in robots]
        return assignment, 0

    # Base case: there is only one task
    if len(tasks) == 1:
        task_id = tasks[0].get_id()
        assignment[task_id] = [robot.get_id() for robot in robots]
        reward = calculate_net_reward(robots, tasks[0])
        return assignment, reward
    
    else:
        # Recursive case: there are multiple tasks
        team_size_0 = partition[0]  # team size for the first task
        task_id = tasks[0].get_id()

        # If no robots assigned to the first task
        if team_size_0 == 0:
            # Create new robot/task lists and sub_partition
            unused_robots = robots.copy()
            other_tasks = tasks[1:]
            sub_partition = partition[1:]

            # Recursively call partition_search
            sub_assignment, add_rewards = partition_search(unused_robots, other_tasks, sub_partition)
            assignment.update(sub_assignment)
            return assignment, add_rewards
        
        # Else, there are some robots assigned to the first task:
        n = len(robots)
        max_reward = float('-inf')
        best_assignment = None
        
        # For all possible combinations of robots for the first task
        combos = combinations(range(n), team_size_0)
        for combo in combos:
            current_assignment = {-1: [], task_id: []}
            for task in tasks[1:]:
                current_assignment[task.get_id()] = []

            unused_robots = robots.copy()
            robots_t0 = [] # These are the robots assigned to the first task

            # Remove robots from unused_robots and add them to robots_t0
            for robot_idx in sorted(combo, reverse=True):
                robot = unused_robots.pop(robot_idx)
                robots_t0.append(robot)
                current_assignment[task_id].append(robot.get_id())

            reward_0 = calculate_net_reward(robots_t0, tasks[0])
            sub_assignment, add_rewards = partition_search(unused_robots, tasks[1:], partition[1:])
            
            # Merge sub_assignment into current_assignment
            for key, value in sub_assignment.items():
                if key in current_assignment:
                    current_assignment[key].extend(value)
                else:
                    current_assignment[key] = value

            reward = reward_0 + add_rewards
            
            if reward > max_reward:
                max_reward = reward
                best_assignment = current_assignment
                
        return best_assignment, max_reward

"""
This function is a modified version of partition_search that allows for some
robots to be assigned to a dummy task (unassigned robots). It returns a dictionary
assignment where -1 represents unassigned robots and task IDs are keys for assigned robots.
"""    
def partition_search_dummyTask(robots, tasks, partition):

    # Error catching
    if len(partition) == 1:
        print("Error: This function should not be called with only one group of robots")
        return None, None
    
    else:
        team_size_unassigned = partition[0]  # team size for the dummy task (unassigned)

        # Initialize the assignment dictionary
        assignment = {-1: []}  # Start with empty list for unassigned robots
        for task in tasks:
            assignment[task.get_id()] = []  # Empty list for each task

        # If no robots assigned to the dummy task
        if team_size_unassigned == 0:
            # Create new robot/task lists and sub_partition
            unused_robots = robots.copy()
            sub_partition = partition[1:]

            # Call partition_search on the non-dummy tasks
            sub_assignment, max_reward = partition_search(unused_robots, tasks, sub_partition)
            assignment.update(sub_assignment)
            return assignment, max_reward

        # Else, there are some robots assigned to the dummy task
        n = len(robots)
        max_reward = float('-inf')
        best_assignment = None
        
        # For all possible combinations of unassigned robots
        combos = combinations(range(n), team_size_unassigned)
        for combo in combos:
            current_assignment = {-1: []}
            for task in tasks:
                current_assignment[task.get_id()] = []

            unused_robots = robots.copy()
    
            # Remove robots from unused_robots and add them to unassigned
            for robot_idx in sorted(combo, reverse=True):
                robot = unused_robots.pop(robot_idx)
                current_assignment[-1].append(robot.get_id())
    
            sub_partition = partition[1:]
            sub_assignment, add_rewards = partition_search(unused_robots, tasks, sub_partition)
            
            # Merge sub_assignment into current_assignment
            for key, value in sub_assignment.items():
                if key in current_assignment:
                    current_assignment[key].extend(value)
                else:
                    current_assignment[key] = value

            reward = add_rewards
            
            if reward > max_reward:
                max_reward = reward
                best_assignment = current_assignment
                
        return best_assignment, max_reward