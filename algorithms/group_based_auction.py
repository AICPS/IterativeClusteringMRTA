import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from itertools import combinations
import phase2.phase2_utils as phase2_utils


def remove_teams_with_robots(task_team_data, robot_ids, task_id=None):
    """
    Remove all teams containing one or more of the specified robots from the data structure.
    
    Args:
        task_team_data: Dictionary {task_id: {team_tuple: (reward, bid)}}
        robot_ids: Single robot ID or tuple/list of robot IDs to remove from teams
        task_id: Specific task to update (None for all tasks)
    """
    if isinstance(robot_ids, int):
        robot_ids = (robot_ids,)
    elif isinstance(robot_ids, list):
        robot_ids = tuple(robot_ids)
    
    robot_set = set(robot_ids)
    
    tasks_to_update = [task_id] if task_id is not None else task_team_data.keys()
    
    for tid in tasks_to_update:
        teams_to_remove = [team for team in task_team_data[tid].keys()
                           if robot_set.intersection(set(team))]
        for team in teams_to_remove:
            del task_team_data[tid][team]


def recalculate_marginal_bids(task_id, assignment, robot_list, task_list, task_team_data, L, printout=False):
    """
    Recalculate marginal bids for a task that already has some robots assigned.
    """
    # Clear existing entries for this task
    task_team_data[task_id] = {}

    current_team = tuple(sorted(assignment[task_id]))
    current_team_size = len(current_team)

    if current_team_size == 0:
        return

    current_robot_team = [robot_list[robot_id] for robot_id in current_team]
    current_reward = phase2_utils.calculate_net_reward(current_robot_team, task_list[task_id])

    if printout:
        print(f"    Recalculating marginal bids for task {task_id}")
        print(f"    Current team: {current_team}, Current reward: {current_reward:.4f}")

    available_robots = assignment[-1].copy()
    max_additional_robots = L - current_team_size

    if max_additional_robots > 0 and available_robots:
        for additional_size in range(1, min(max_additional_robots + 1, len(available_robots) + 1)):
            for additional_team in combinations(available_robots, additional_size):
                combined_team = current_team + additional_team
                combined_robot_team = [robot_list[rid] for rid in combined_team]

                combined_reward = phase2_utils.calculate_net_reward(combined_robot_team, task_list[task_id])
                marginal_reward = combined_reward - current_reward

                if marginal_reward > 0:
                    marginal_bid = marginal_reward / len(additional_team)
                    task_team_data[task_id][additional_team] = (marginal_reward, marginal_bid)


def group_based_auction(robot_list, task_list, hypes, printout=False):
    start_time = time.time()
    L = hypes['L']
    num_tasks = len(task_list)
    num_robots = len(robot_list)

    total_reward = 0
    assignment = {-1: list(range(num_robots))}
    assignment.update({task_id: [] for task_id in range(num_tasks)})

    checkpoint_rewards = []
    checkpoint_times = []
    round_number = 0

    # Generate all possible teams up to size L
    all_teams = []
    for team_size in range(1, min(L + 1, num_robots + 1)):
        for team in combinations(range(num_robots), team_size):
            all_teams.append(team)

    if printout:
        print(f"Generated {len(all_teams)} possible teams of size 1 to {min(L, num_robots)}")
        print(f"Time to create teams: {time.time() - start_time:.4f} seconds")

    # Initialize single dictionary for all tasks
    task_team_data = {task_id: {} for task_id in range(num_tasks)}

    # Initial calculation of reward and bid
    for task_id in range(num_tasks):
        task = task_list[task_id]
        for team in all_teams:
            robot_team = [robot_list[rid] for rid in team]
            net_reward = phase2_utils.calculate_net_reward(robot_team, task)
            if net_reward > 0:
                bid = net_reward / len(team)
                task_team_data[task_id][team] = (net_reward, bid)

        if printout:
            positive_teams = len(task_team_data[task_id])
            print(f"Task {task_id}: Calculated rewards and bids for {positive_teams} teams")
            print(f"Time elapsed: {time.time() - start_time:.4f} seconds")

    # Main auction loop
    while True:
        round_number += 1
        if printout:
            print(f"\nROUND {round_number}")
            print(f"Current assignments: {assignment}")
            print(f"Unassigned robots: {assignment[-1]}")
            print(f"Current total reward: {total_reward:.4f}")

        # Find max bid
        max_bid = -1
        winning_task_id = None
        winning_team = None
        for task_id, teams in task_team_data.items():
            for team, (reward, bid) in teams.items():
                if bid > max_bid:
                    max_bid = bid
                    winning_task_id = task_id
                    winning_team = team

        if winning_task_id is None or winning_team is None or max_bid <= 0:
            if printout:
                print("\nNo more positive bids found - auction complete!")
                print(f"Final assignments: {assignment}")
                print(f"Unassigned robots: {assignment[-1]}")
                print(f"Final total reward: {total_reward:.4f}")
            break

        if printout:
            print(f"Winning task: {winning_task_id}, Winning team: {winning_team}, Bid: {max_bid:.4f}")

        # Assign robots
        for rid in winning_team:
            if rid in assignment[-1]:
                assignment[-1].remove(rid)
            assignment[winning_task_id].append(rid)

        round_reward, _ = task_team_data[winning_task_id][winning_team]
        total_reward += round_reward

        if printout:
            print(f"Updated assignment for task {winning_task_id}: {assignment[winning_task_id]}")
            print(f"Remaining unassigned robots: {assignment[-1]}")
            print(f"New total reward: {total_reward:.4f}")

        checkpoint_rewards.append(total_reward)
        checkpoint_times.append(time.time() - start_time)

        # Remove teams containing assigned robots
        if printout:
            print(f"Removing teams containing robots {winning_team} from all tasks...")
        remove_teams_with_robots(task_team_data, winning_team)

        # Recalculate marginal bids for winning task
        recalculate_marginal_bids(winning_task_id, assignment, robot_list, task_list,
                                  task_team_data, L, printout)

        if printout:
            remaining_bids = sum(len(teams) for teams in task_team_data.values())
            print(f"Remaining total bids across all tasks: {remaining_bids}")

    checkpoint_rewards.append(total_reward)
    checkpoint_times.append(time.time() - start_time)

    if printout:
        print("\nAUCTION SUMMARY:")
        print(f"Total rounds: {round_number - 1}")
        print(f"Final assignments: {assignment}")
        print(f"Unassigned robots: {assignment[-1]} (count: {len(assignment[-1])})")
        print(f"Final total reward: {total_reward:.4f}")
        print(f"Total time: {time.time() - start_time:.4f} seconds")

    return assignment, total_reward, checkpoint_rewards, checkpoint_times