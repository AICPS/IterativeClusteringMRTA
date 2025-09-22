import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import math
import phase2.phase2_utils as phase2_utils

def SA_efficient(robot_list, task_list, hypes, time_limit = 5.0, printout=False):
    """
    Simulated Annealing algorithm for task assignment using team-based representation.
    Each solution is represented as a dictionary where:
    - Keys are task IDs where -1 represents unassigned
    - Values are lists of robot IDs assigned to that task
    """
    # Track Time
    mutate_time = 0
    reward_time = 0
    start_time = time.time()

    # Required Hyperparameters
    #num_solutions_to_test = hypes['SA_solutions_to_test']
    initial_temp = hypes['initial_SA_temp']
    alpha = initial_temp/time_limit
    
    L = hypes['L']  # Max robots per task
    num_tasks = len(task_list)
    num_robots = len(robot_list)

    best_reward = 0
    best_assignment = {}
    checkpoint_rewards = []
    checkpoint_times = []

    # Create a initial team assignment (All unassigned)
    current_solution = {}
    
    # Store rewards for each team
    team_rewards = {task_id: 0 for task_id in current_solution if task_id != -1}
    
    # Initialize empty teams dictionary with all tasks and unassigned
    current_solution = {-1: [robot_id for robot_id in range(num_robots)]}  # Unassigned robots
    for task_id in range(num_tasks):
        current_solution[task_id] = []
    
    # Calculate initial reward
    current_reward = 0
    for task_id in range(num_tasks):
        robot_team = [robot_list[robot_id] for robot_id in current_solution[task_id]]
        task = task_list[task_id]
        team_reward = phase2_utils.calculate_net_reward(robot_team, task)
        team_rewards[task_id] = team_reward
        current_reward += team_reward
    
    best_reward = current_reward
    best_assignment = current_solution.copy()

    # checkpoint_rewards.append(best_reward)
    # checkpoint_times.append(time.time() - start_time)

    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        new_solution, changed_tasks = team_mutate_single(current_solution, num_robots, L)
        
        # Calculate reward of new solution
        delta_reward = calculate_delta_reward(new_solution, changed_tasks, robot_list, task_list, team_rewards)
        new_reward = delta_reward + current_reward
        
        # Update best assignments/rewards
        if new_reward > best_reward:
            best_reward = new_reward
            best_assignment = new_solution.copy()
        
        # Update current solution/reward
        if delta_reward > 0:
            current_solution = new_solution
            current_reward = new_reward
            # Update team_rewards for changed tasks
            for task_id in changed_tasks:
                if task_id != -1:
                    robot_team = [robot_list[robot_id] for robot_id in new_solution[task_id]]
                    task = task_list[task_id]
                    team_rewards[task_id] = phase2_utils.calculate_net_reward(robot_team, task)
        else:
            temp = max(initial_temp - alpha*(time.time()-start_time), 1e-6)
            prob_accept_worse = math.exp(delta_reward/temp)
            if random.random() < prob_accept_worse:
                current_solution = new_solution
                current_reward = new_reward
                # Update team_rewards for changed tasks
                for task_id in changed_tasks:
                    if task_id != -1:
                        robot_team = [robot_list[robot_id] for robot_id in new_solution[task_id]]
                        task = task_list[task_id]
                        team_rewards[task_id] = phase2_utils.calculate_net_reward(robot_team, task)

        if iteration % 100 == 0:
            # print(f'Checkpoint Reached!')
            # print(f'Best Reward: {best_reward}')
            # print(f'Current Reward: {current_reward}')
            #print(f'Best Team Assignment: {best_assignment}')
            checkpoint_rewards.append(best_reward)
            checkpoint_times.append(time.time() - start_time)
    
    # print(f"Reward Time: {reward_time}")
    # print(f"Mutate Time: {mutate_time}")
    return best_reward, best_assignment, checkpoint_rewards, checkpoint_times

def calculate_delta_reward(new_solution, changed_tasks, robot_list, task_list, old_team_rewards):
    delta_reward = 0
    for task_id in changed_tasks:
        if task_id != -1:
            old_reward = old_team_rewards[task_id]
            robot_team = [robot_list[robot_id] for robot_id in new_solution[task_id]]
            task = task_list[task_id]
            new_team_reward = phase2_utils.calculate_net_reward(robot_team, task)
            delta_reward += new_team_reward - old_reward
    return delta_reward

def calculate_assignment_reward(team_assignment, robot_list, task_list):
    """
    Calculate the reward directly from a team assignment dictionary.
    """
    total_reward = 0
    
    # Sum the rewards of each team
    for task_id in range(len(task_list)):
        if len(team_assignment[task_id]) > 0:
            # Convert robot IDs to robot objects
            robot_team = [robot_list[robot_id] for robot_id in team_assignment[task_id]]
            task = task_list[task_id]
            reward = phase2_utils.calculate_net_reward(robot_team, task)
            total_reward += reward
            
    return total_reward

# Single Task Mutation
def team_mutate_single(team_assignment, num_robots, L):
    """
    Mutate a team assignment by randomly modifying one team.
    """
    # Create a copy of the team assignment
    mutated_teams = {task_id: list(team) for task_id, team in team_assignment.items()}
    
    # Select a random task to mutate (excluding unassigned)
    task_to_mutate = random.randint(0, len(mutated_teams) - 2)  # -2 because we exclude -1
    
    # Clear the team we're mutating
    mutated_teams[task_to_mutate] = []
    
    # Decide on a new team size
    new_team_size = random.randint(1, min(L, num_robots))
    
    # Track which teams lost robots (list of task ids)
    teams_to_refill = []
    
    if new_team_size > 0:
        # Randomly select robots for the new team
        new_team_robots = random.sample(range(num_robots), new_team_size)
        
        # For each selected robot, remove it from its current team
        for robot_id in new_team_robots:
            # Find and remove the robot from its current team
            for task_id, team in mutated_teams.items():
                if robot_id in team:
                    team.remove(robot_id)
                    # Add team to refill list (except unassigned or the mutated task)
                    if task_id != -1 and task_id != task_to_mutate:
                        teams_to_refill.append(task_id)
                    break
        
        # Assign the new team
        mutated_teams[task_to_mutate] = new_team_robots
        
        # Collect all unassigned robots
        all_assigned = set()
        for task_id, team in mutated_teams.items():
            all_assigned.update(team)
        unassigned_robots = list(set(range(num_robots)) - all_assigned)
        random.shuffle(unassigned_robots)
        
        # Try to refill teams that lost robots
        for task_id in teams_to_refill:
            if unassigned_robots:
                # Add one robot to this team
                robot_id = unassigned_robots.pop(0)
                mutated_teams[task_id].append(robot_id)
        
        # Update unassigned pool
        mutated_teams[-1] = unassigned_robots
    
    return mutated_teams, set(teams_to_refill + [task_to_mutate])  # Return changed tasks as well