import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot

"""Task Benefit Functions for Generating Problems with 2 different robot types"""
def generate_task_types_2_capabilities(L, kappa):
    # Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
    reward_dim = tuple(L+1 for _ in range(kappa))
    task_types = {
        0: np.zeros(reward_dim),  # Type 0: can be done by robots with capability 1
        1: np.zeros(reward_dim),  # Type 1: can be done by robots with capability 2
        2: np.zeros(reward_dim),  # Type 2: can be done only collaboratively by cap 1 and 2
        3: np.zeros(reward_dim),  # Type 3: can be done only collaboratively by two of cap 1
        4: np.zeros(reward_dim),  # Type 4: can be done only collaboratively by two of cap 2
        5: np.zeros(reward_dim),  # Type 5: can be done only collaboratively by cap 1 and 2, diminishing returns
    }

    # Type 0
    task_types[0][1,0] = 100
    task_types[0][2,0] = 150
    task_types[0][3,0] = 200

    # Type 1
    task_types[1][0,1] = 100
    task_types[1][0,2] = 150
    task_types[1][0,3] = 200

    # Type 2
    task_types[2][2,0] = 200

    # Type 3
    task_types[3][0,2] = 200

    # Type 4
    task_types[4][1,1] = 200
    task_types[4][1,2] = 220
    task_types[4][2,1] = 220

    # Type 5
    task_types[5][1,2] = 350
    task_types[5][2,1] = 350

    return task_types

"""Task Benefit Functions for Generating Problems with 3 different robot types"""
def generate_task_types_3_capabilities(L, kappa):
    # Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
    reward_dim = tuple(L+1 for _ in range(kappa))
    print(f"reward_dim {reward_dim}")

    task_types = {
        0: np.zeros(reward_dim),  # Type 0: can be done by robots with capability 1
        1: np.zeros(reward_dim),  # Type 1: can be done by robots with capability 2
        2: np.zeros(reward_dim),  # Type 2: can be done by robots with capability 3
        3: np.zeros(reward_dim),  # Type 3: can be done only collaboratively by two of cap 1
        4: np.zeros(reward_dim),  # Type 4: can be done only collaboratively by two of cap 2
        5: np.zeros(reward_dim),  # Type 5: can be done only collaboratively by two of cap 3
        6: np.zeros(reward_dim),  # Type 6: requires types 1 and 2
        7: np.zeros(reward_dim),  # Type 7: requires types 1 and 3
        8: np.zeros(reward_dim),  # Type 8: requires types 2 and 3
        9: np.zeros(reward_dim),  # Type 9: requires types 1, 2, and 3
    }

    # Type 0 Marginally Diminishing Returns
    task_types[0][1,0,0] = 100
    task_types[0][2,0,0] = 150
    task_types[0][3,0,0] = 175

    # Type 1 Marginally Diminishing Returns
    task_types[1][0,1,0] = 100
    task_types[1][0,2,0] = 150
    task_types[1][0,3,0] = 175

    # Type 2 Marginally Diminishing Returns
    task_types[2][0,0,1] = 100
    task_types[2][0,0,2] = 150
    task_types[2][0,0,3] = 175

    # Type 3 Marginally Increasing Returns
    task_types[3][1,0,0] = 50
    task_types[3][2,0,0] = 200

    # Type 4 Marginally Increasing Returns
    task_types[4][0,1,0] = 50
    task_types[4][0,2,0] = 200

    # Type 5 Marginally Increasing Returns
    task_types[5][0,0,1] = 50
    task_types[5][0,0,2] = 200

    # Type 6 Linear Returns
    task_types[6][1,0,0] = 50
    task_types[6][0,1,0] = 50
    task_types[6][1,1,0] = 100

    # Type 7 Linear Returns
    task_types[7][1,0,0] = 50
    task_types[7][0,0,1] = 50
    task_types[7][1,0,1] = 100

    # Type 8 Linear Returns
    task_types[8][0,1,0] = 50
    task_types[8][0,0,1] = 50
    task_types[8][0,1,1] = 100

    # Type 9 Threshold Returns
    task_types[9][1,1,1] = 300

    return task_types

""" Creates a problem instance with 3 different robot types.
    Robots are evenly distributed among the 3 types.
    Tasks are evenly distributed among the 10 different task types.
    Robots and tasks are randomly placed in a 2D space defined by max_x and max_y."""
def generate_problem_3_capabilities(hypes, max_x, max_y):

    robot_type_1 = [1,0,0]
    robot_type_2 = [0,1,0]
    robot_type_3 = [0,0,1]
    min_x = 0
    min_y = 0
    nu = hypes['nu']  # number of robots
    mu = hypes['mu']  # number of tasks
    kappa = hypes['kappa']  # number of capabilities
    L = hypes['L']  # maximum team size



    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    for i in range(nu):
        if i % 3 == 0:
            robot_type = robot_type_1
        elif i % 3 == 1:
            robot_type = robot_type_2
        else:
            robot_type = robot_type_3
        
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types_3_capabilities(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = i % len(task_types)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

""" Creates a problem instance with 2 different robot types.
    Robots are evenly distributed among the 2 types.
    Tasks are evenly distributed among the 10 different task types.
    Robots and tasks are randomly placed in a 2D space defined by max_x and max_y."""
def generate_problem_2_capabilities(hypes, max_x, max_y):

    robot_type_1 = [1,0]
    robot_type_2 = [0,1]
    min_x = 0
    min_y = 0
    nu = hypes['nu']  # number of robots
    mu = hypes['mu']  # number of tasks
    kappa = hypes['kappa']  # number of capabilities
    L = hypes['L']  # maximum team size



    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    for i in range(nu):
        if i % 2 == 0:
            robot_type = robot_type_1
        else:
            robot_type = robot_type_2
        
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types_2_capabilities(L, kappa)
    for i in range(mu):
        task_type_key = i % len(task_types)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

""" Creates a problem instance with 2 different robot types.
    Robots randomly chosen to be one of 3 types.
    Tasks are randomly chosen to be one of 10 different task types.
    Robots and tasks are randomly placed in a 2D space defined by max_x and max_y.
    # Note the key difference between this and generate_problem_3_capabilities is that
    # here the robots and tasks are randomly assigned types, whereas in the other function
    # they are evenly distributed among the types."""
def generate_random_problem_instance_3(hypes, max_x, max_y):
    
    robot_type_1 = [1,0,0]
    robot_type_2 = [0,1,0]
    robot_type_3 = [0,0,1]
    min_x = 0
    min_y = 0
    nu = hypes['nu']  # number of robots
    mu = hypes['mu']  # number of tasks
    kappa = hypes['kappa']  # number of capabilities
    L = hypes['L']  # maximum team size

    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    for i in range(nu):
        r = random.random()
        if r <= 0.33:
            robot_type = robot_type_1
        elif r <= 0.66:
            robot_type = robot_type_2
        else:
            robot_type = robot_type_3
        
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types_3_capabilities(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

"Prints out the details of a problem instance"
def print_problem_instance(robot_list, task_list):
    print(f"Created {len(robot_list)} robots:")
    for robot in robot_list:
        print(f"Robot {robot.id}: Position ({robot.x}, {robot.y})")
        print(f"Capabilities: {robot.capabilities}")
        print()  # Add an empty line for better readability

    print(f"\nCreated {len(task_list)} tasks:")
    for task in task_list:
        print(f"Task {task.id}: Position ({task.x}, {task.y})")
        print(f"Reward Matrix:\n{task.reward_matrix}")
        print()  # Add an empty line for better readability