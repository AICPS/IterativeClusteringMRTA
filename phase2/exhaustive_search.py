import numpy as np
import math
from itertools import combinations
from shared_classes.robot import Robot

""" 
This code performs an exhaustive search for the optimal assignment of robots to tasks.
Used to check that IP_assignment correctly finds optimal assignments.
"""

# My Code:
import phase2.phase2_utils as utils

def exhaustive_search(robot_list, task_list, L,kappa,printout=False):
    
    n = len(robot_list)
    m = len(task_list)

    # Check if there are no tasks or robots
    if n == 0:
        best_assignment = [ [] for _ in range(m+1)]
        reward = 0
        return best_assignment, reward
    if m == 0:
        best_assignment = [[robot.get_id() for robot in robot_list]]
        reward = 0
        return best_assignment, reward
    

    best_assignment = None
    global_reward = float('-inf')

    partitions = utils.generate_partitions(n,m,L)

    if printout:
        print(f"All partitions of {n} into {m}+1 parts with first part unassigned:")
        for partition in partitions:
            print(partition)
        print(f"Total number of partitions: {len(partitions)}")


    while len(partitions) > 0:
        
        partition_idx = 0
        
        # Search partition
        partition_assignment, partition_reward = utils.partition_search_dummyTask(robot_list, task_list, partitions[partition_idx])
        
        # Remove best partition from list
        partitions.pop(partition_idx)
        
        if partition_reward > global_reward:
            global_reward = partition_reward
            best_assignment = partition_assignment
            
        if printout:
            print("\nbest_assignment: ", best_assignment)
            print("global_reward: ", global_reward)
            print("num partitions left: ", len(partitions))
        
    if printout:
        print("Final best_assignment: ", best_assignment)
        print("Final global_reward: ", round(global_reward))
    
    return best_assignment, global_reward