import numpy as np

class Cluster:
    def __init__(self, robot_ids, task_ids, robot_list, task_list,kappa):
        self.robot_ids = robot_ids
        self.task_ids = task_ids
        self.num_robots = len(robot_ids)
        self.num_tasks = len(task_ids)
        
        # Extract robots and tasks based on ids
        self.robots = [robot_list[id] for id in robot_ids]
        self.tasks = [task_list[id] for id in task_ids]
        
        # Calculate capability vector
        if len(self.robots) == 0:
            self.capability_vector = np.zeros(kappa, dtype=np.int32)
        else:
            self.capability_vector = np.sum([robot.get_capabilities() for robot in self.robots], axis=0).astype(np.int32)
        
        # Calculate task requirement vector
        if len(self.tasks) == 0:
            self.task_requirement_vector = np.zeros(kappa, dtype=np.int32)
        else:
            self.task_requirement_vector = np.sum([task.get_capability_vector() for task in self.tasks], axis=0).astype(np.int32)
        
        # Calculate center of mass
        robot_locations = [robot.get_location() for robot in self.robots]
        task_locations = [task.get_location() for task in self.tasks]
        all_locations = robot_locations + task_locations
        self.center_of_mass = np.mean(all_locations, axis=0)
    
    def get_center_of_mass(self):
        return self.center_of_mass

    def __str__(self):
        return (f"Cluster with {self.num_robots} robots and {self.num_tasks} tasks\n"
                f"Robot IDs: {self.robot_ids}\n"
                f"Task IDs: {self.task_ids}\n"
                f"Capability Vector: {self.capability_vector}\n"
                f"Task Requirement Vector: {self.task_requirement_vector}\n"
                f"Center of Mass: {self.center_of_mass}")
        
    def merge(self, other_cluster):
        # Merge robot and task IDs
        self.robot_ids.extend(other_cluster.robot_ids)
        self.task_ids.extend(other_cluster.task_ids)
        
        # Update counts
        self.num_robots = len(self.robot_ids)
        self.num_tasks = len(self.task_ids)
        
        # Merge robots and tasks
        self.robots.extend(other_cluster.robots)
        self.tasks.extend(other_cluster.tasks)
        
        # Update capability vector
        self.capability_vector = (self.capability_vector + other_cluster.capability_vector).astype(np.int64)
        
        # Update task requirement vector
        self.task_requirement_vector = (self.task_requirement_vector + other_cluster.task_requirement_vector).astype(np.int64)
        
        # Recalculate center of mass by weighted average
        total_entities = self.num_robots + self.num_tasks + other_cluster.num_robots + other_cluster.num_tasks
        self.center_of_mass = (
            (self.center_of_mass * (self.num_robots + self.num_tasks) + 
                other_cluster.center_of_mass * (other_cluster.num_robots + other_cluster.num_tasks)
            ) / total_entities
        )