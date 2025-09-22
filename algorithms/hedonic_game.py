import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import math
from itertools import combinations
import phase2.phase2_utils as phase2_utils

import sys
import os
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase2.phase2_utils as phase2_utils

def hedonic_game(robot_list, task_list, hypes, time_limit=10, printout=False):
    start_time = time.time()
    
    # Hyperparameters
    L = hypes['L']  # Max robots per task
    num_tasks = len(task_list)
    num_robots = len(robot_list)

    # Initial assignment: all robots unassigned
    assignment = {-1: list(range(num_robots))}
    assignment.update({task_id: [] for task_id in range(num_tasks)})

    # Initialize reward tracking
    checkpoint_rewards = []
    checkpoint_times = []

    total_reward = 0.0
    equilibrium = False
    iteration = 0

    while not equilibrium and (time.time() - start_time) < time_limit:
        equilibrium = True
        iteration += 1

        robot_order = list(range(num_robots))
        random.shuffle(robot_order)

        for robot_id in robot_order:
            # Identify current task of robot
            current_task = next(
                (tid for tid, robots in assignment.items() if robot_id in robots), -1)

            # Calculate marginal utility for current task
            if current_task == -1:
                current_marginal_utility = 0.0
            else:
                current_team_ids = assignment[current_task]
                current_team = [robot_list[rid] for rid in current_team_ids]
                
                if len(current_team_ids) == 1:
                    current_reward = phase2_utils.calculate_net_reward(current_team, task_list[current_task])
                    reward_without_robot = 0.0
                else:
                    current_reward = phase2_utils.calculate_net_reward(current_team, task_list[current_task])
                    team_without_ids = [rid for rid in current_team_ids if rid != robot_id]
                    team_without = [robot_list[rid] for rid in team_without_ids]
                    reward_without_robot = phase2_utils.calculate_net_reward(team_without, task_list[current_task])

                current_marginal_utility = current_reward - reward_without_robot

            # Initialize best option
            best_task = current_task
            best_marginal_utility = current_marginal_utility

            # Option: remain unassigned
            if 0.0 > best_marginal_utility:
                best_task = -1
                best_marginal_utility = 0.0

            # Evaluate all other tasks
            for task_id in range(num_tasks):
                if task_id == current_task:
                    continue

                team_ids = assignment[task_id]
                if len(team_ids) >= L:
                    continue  # Task full

                new_team_ids = team_ids + [robot_id]
                new_team = [robot_list[rid] for rid in new_team_ids]
                current_team = [robot_list[rid] for rid in team_ids]

                reward_with_robot = phase2_utils.calculate_net_reward(new_team, task_list[task_id])
                reward_without_robot = phase2_utils.calculate_net_reward(current_team, task_list[task_id])
                marginal_utility = reward_with_robot - reward_without_robot

                if marginal_utility > best_marginal_utility:
                    best_marginal_utility = marginal_utility
                    best_task = task_id

            # Move robot if it improves utility
            if best_task != current_task:
                equilibrium = False  # Movement occurred

                # Remove from old task
                assignment[current_task].remove(robot_id)

                # Add to new task
                assignment[best_task].append(robot_id)

                # Update total reward incrementally
                total_reward += (best_marginal_utility - current_marginal_utility)

                if printout:
                    print(f"Robot {robot_id} moved from task {current_task} to {best_task} "
                          f"(Δ = {best_marginal_utility - current_marginal_utility:.3f})")

        # Record checkpoint
        checkpoint_rewards.append(total_reward)
        checkpoint_times.append(time.time() - start_time)

        if printout:
            print(f"Iteration {iteration}: Total reward = {total_reward:.3f}, "
                  f"Time elapsed = {checkpoint_times[-1]:.3f}s")

    # Final reward recomputation (sanity check or debug)
    final_total_reward = 0.0
    for task_id in range(num_tasks):
        if assignment[task_id]:
            team = [robot_list[rid] for rid in assignment[task_id]]
            final_total_reward += phase2_utils.calculate_net_reward(team, task_list[task_id])

    if printout:
        print(f"Hedonic game completed after {iteration} iterations")
        print(f"Final total reward: {final_total_reward:.3f}")
        print(f"Equilibrium reached: {equilibrium}")
        print(f"Final assignment: {assignment}")

    return assignment, final_total_reward, checkpoint_rewards, checkpoint_times


