import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot

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

def generate_task_types_2_capabilities_mdr(L, kappa):
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
    task_types[0][3,0] = 175

    # Type 1
    task_types[1][0,1] = 100
    task_types[1][0,2] = 150
    task_types[1][0,3] = 175

    # Type 2
    task_types[2][1,0] = 50
    task_types[2][2,0] = 75
    task_types[2][1,1] = 100
    task_types[2][2,1] = 125

    # Type 3
    task_types[3][0,1] = 50
    task_types[3][0,2] = 75
    task_types[3][1,1] = 100
    task_types[3][1,2] = 125

    # Type 4
    task_types[4][0,1] = 50
    task_types[4][1,0] = 50
    task_types[4][1,1] = 100

    # Type 5
    task_types[5][0,1] = 100
    task_types[5][1,0] = 100
    task_types[5][1,1] = 150
    task_types[5][1,2] = 200
    task_types[5][2,1] = 200

    return task_types

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

def generate_task_types_3_capabilities_mdr(L, kappa):
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

    # Type 0
    task_types[0][1,0,0] = 100
    task_types[0][2,0,0] = 150
    task_types[0][3,0,0] = 175

    # Type 1
    task_types[1][0,1,0] = 100
    task_types[1][0,2,0] = 150
    task_types[1][0,3,0] = 175

    # Type 2
    task_types[2][0,0,1] = 100
    task_types[2][0,0,2] = 150
    task_types[2][0,0,3] = 175

    # Type 3
    task_types[3][1,0,0] = 100
    task_types[3][0,1,0] = 100
    task_types[3][0,0,1] = 100
    task_types[3][1,1,0] = 150
    task_types[3][1,0,1] = 150
    task_types[3][0,1,1] = 150
    task_types[3][1,1,1] = 200

    # Type 4
    task_types[4][1,0,0] = 100
    task_types[4][0,1,0] = 100
    task_types[4][1,1,0] = 200

    # Type 5
    task_types[5][1,0,0] = 100
    task_types[5][0,0,1] = 100
    task_types[5][1,0,1] = 200

    # Type 6
    task_types[6][0,1,0] = 100
    task_types[6][0,0,1] = 100
    task_types[6][0,1,1] = 200

    # Type 7
    task_types[7][1,0,0] = 50
    task_types[7][2,0,0] = 75
    task_types[7][1,1,0] = 100
    task_types[7][2,1,0] = 150

    # # Type 8
    task_types[8][0,0,1] = 50
    task_types[8][0,0,2] = 75
    task_types[8][0,1,1] = 100
    task_types[8][0,1,2] = 150

    # # Type 9
    task_types[9][0,1,0] = 50
    task_types[9][0,2,0] = 75
    task_types[9][1,1,0] = 100
    task_types[9][1,2,0] = 125

    return task_types

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

def generate_random_problem_instance_3_mdr(hypes, max_x, max_y):
    
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
    task_types = generate_task_types_3_capabilities_mdr(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

def generate_random_problem_instance_2(hypes, max_x, max_y, prob_type1=0.5):
    
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
        if random.random() < prob_type1:
            robot_type = robot_type_1
        else:
            robot_type = robot_type_2
        
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types_2_capabilities(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

def generate_random_problem_instance_2_mdr(hypes, max_x, max_y, prob_type1=0.5):
    
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
        if random.random() < prob_type1:
            robot_type = robot_type_1
        else:
            robot_type = robot_type_2
        
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types_2_capabilities_mdr(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

def generate_problem_instance_discrete_task_requirements(hypes, max_x, max_y):
    
    # Tasks will either take type 1 or type 2 robots, never both.

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
        robot_type =random.choice([robot_type_1, robot_type_2])       
        robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types(L, kappa)
    # Remove types
    task_types.pop(9)
    task_types.pop(8)
    task_types.pop(5)
    task_types.pop(2)
    print(task_types)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

def generate_problem_instance_individual_tasks(hypes, max_x, max_y):
    
    # Tasks will only have positive reward for team size of 1
    # Only tasks 6,7,8

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
        robot_type =random.choice([robot_type_1, robot_type_2])       
        robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types(L, kappa)
    # Remove collaborative tasks
    task_types.pop(9)
    task_types.pop(5)
    task_types.pop(4)
    task_types.pop(3)
    task_types.pop(2)
    task_types.pop(1)
    task_types.pop(0)
    print(task_types)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i], kappa)
        task_list.append(task)

    return robot_list, task_list

def generate_problem_instance_homogeneous_robots(hypes, max_x, max_y):
    
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
        robot_type = robot_type_1      
        robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i])
        task_list.append(task)

    return robot_list, task_list

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