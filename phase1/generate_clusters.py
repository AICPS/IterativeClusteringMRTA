"""This is a combination of spatial clustering and capability clustering."""

import sys
import os
import random
import numpy as np
import math
from scipy.spatial.distance import euclidean
from shared_classes.cluster import Cluster  # Import the Cluster class

def convert_assignment_to_clusters(assignment, robot_list, task_list,kappa):
    """
    Convert the assignment to a list of Cluster objects.

    Args:
    assignment (dict): A dictionary where keys are task IDs (and -1 for unassigned),
                       and values are lists of robot IDs assigned to each task.
    robot_list (list): A list of all Robot objects.
    task_list (list): A list of all Task objects.

    Returns:
    list: A list of Cluster objects.
    """
    
    clusters = []

    # Handle unassigned robots (key -1)
    unassigned_robots = assignment.get(-1, [])
    for robot_id in unassigned_robots:
        # Create a cluster with just the robot
        clusters.append(Cluster([robot_id], [], robot_list, task_list,kappa))

    # Handle assigned tasks and robots
    for task_id, robot_ids in assignment.items():
        if task_id != -1:  # Skip the unassigned robots key
            clusters.append(Cluster(robot_ids, [task_id], robot_list, task_list,kappa))
                
    return clusters

def temp_softmax(distances, temperature):
    exps = np.exp(-np.array(distances) / temperature)
    return exps / exps.sum()

def capability_match_filter(cluster1, cluster2):
    """
    Calculate the capability match score between two clusters.
    """

    dot_product12 = np.dot(cluster1.capability_vector, cluster2.task_requirement_vector)
    dot_product21 = np.dot(cluster2.capability_vector, cluster1.task_requirement_vector)

    # print(f"cluster1 capability vector: {cluster1.capability_vector}")
    # print(f"cluster2 task requirement vector: {cluster2.task_requirement_vector}")
    # print(f"cluster2 capability vector: {cluster2.capability_vector}")
    # print(f"cluster1 task requirement vector: {cluster1.task_requirement_vector}")

    # print(f"cap_match 1 to 2: {dot_product12}")
    # print(f"cap_match 2 to 1: {dot_product21}")

    # Convert dot products to scalars if they're arrays
    if isinstance(dot_product12, np.ndarray):
        dot_product12 = dot_product12.item()
    if isinstance(dot_product21, np.ndarray):
        dot_product21 = dot_product21.item()

    if dot_product12 == 0 and dot_product21 == 0:
        return 0
    else:
        return 1

def refine_clusters_random_merge(initial_clusters, L_r, L_t):
    #print(f'Random L_r: {L_r}, L_t: {L_t}')

    clusters = initial_clusters.copy()
    
    # Create list of clusters that can still increase in size without exceeding L_r or L_t
    incomplete_clusters = list(range(len(clusters)))
    
    # Create list of clusters that are already at max size
    complete_clusters = []

    while incomplete_clusters:
        # Randomly choose one of the incomplete clusters
        cluster_index = random.choice(incomplete_clusters)
        cluster = clusters[cluster_index]
        
        # Find all possible merge pairs for this cluster that do not exceed L_r or L_t
        merge_candidates = []
        for j in incomplete_clusters:
            if j != cluster_index:
                if clusters[cluster_index].num_robots + clusters[j].num_robots <= L_r and \
                    clusters[cluster_index].num_tasks + clusters[j].num_tasks <= L_t :
                    merge_candidates.append(j)

        if not merge_candidates:
            # If there are no possible merge pairs, move this cluster to the complete_clusters list
            complete_clusters.append(cluster_index)
            incomplete_clusters.remove(cluster_index)
        else:
            # Choose a random merge pair
            merge_cluster_index = random.choice(merge_candidates)
            
            # Perform the merge
            cluster.merge(clusters[merge_cluster_index])
            clusters.pop(merge_cluster_index)
            
            # Remove the merged cluster from incomplete_clusters
            if merge_cluster_index in incomplete_clusters:
                incomplete_clusters.remove(merge_cluster_index)
            
            # Adjust indices in incomplete_clusters
            incomplete_clusters = [i if i < merge_cluster_index else i-1 for i in incomplete_clusters]
    
    return clusters

def refine_clusters_heuristic(initial_clusters, L_r, L_t, temp):
    clusters = initial_clusters.copy()
    
    # Create list of clusters that can still increase in size
    incomplete_clusters = [i for i, cluster in enumerate(clusters)]
    
    while len(incomplete_clusters) > 1:
        # Randomly choose one of the incomplete clusters
        cluster_index = random.choice(incomplete_clusters)
        cluster = clusters[cluster_index]
        
        # Find all possible merge candidates for this cluster
        merge_candidates = []
        for j in incomplete_clusters:
            if j != cluster_index:
                if (len(cluster.robots) + len(clusters[j].robots) <= L_r and 
                    len(cluster.tasks) + len(clusters[j].tasks) <= L_t):
                    merge_candidates.append(j)

        if not merge_candidates:
            # If there are no possible merge pairs, remove this cluster from incomplete_clusters
            incomplete_clusters.remove(cluster_index)
        else:
            # Calculate capability matching scores
            cap_match_scores = np.array([capability_match_filter(cluster, clusters[i]) for i in merge_candidates])
            
            cap_match_sum = np.sum(np.abs(cap_match_scores))
            
            if cap_match_sum == 0: #This means that all capability matching scores are zero
                incomplete_clusters.remove(cluster_index)
            else:
                # Calculate capability matching probabilities
                cap_match_probs = np.abs(cap_match_scores) / cap_match_sum

                # Calculate spatial distances
                distances = [euclidean(cluster.get_center_of_mass(), clusters[j].get_center_of_mass()) 
                             for j in merge_candidates]
            
                # Calculate spatial probabilities
                spatial_probs = temp_softmax(distances, temp)
            
                # Combine probabilities using element-wise product
                combined_probs = spatial_probs * cap_match_probs
            
                # Normalize the combined probabilities
                combined_probs_sum = np.sum(combined_probs)
                if combined_probs_sum == 0:
                    # If all combined probabilities are zero, skip this cluster
                    # This should never happen since softmax probs are non-negative
                    print("Warning: All combined probabilities are zero. Skipping this cluster.")
                    incomplete_clusters.remove(cluster_index)
                else:
                    combined_probs /= combined_probs_sum
            
                    # Choose a merge candidate based on combined probabilities
                    merge_index = np.random.choice(len(merge_candidates), p=combined_probs)
                    merge_cluster_index = merge_candidates[merge_index]
            
                    # Perform the merge
                    cluster.merge(clusters[merge_cluster_index])
                    clusters.pop(merge_cluster_index)
            
                    # Remove the merged cluster from incomplete_clusters
                    if merge_cluster_index in incomplete_clusters:
                        incomplete_clusters.remove(merge_cluster_index)
            
                    # Adjust indices in incomplete_clusters
                    incomplete_clusters = [i if i < merge_cluster_index else i-1 for i in incomplete_clusters]
    
    return clusters

def refine_clusters_cap_match_filter(initial_clusters, L_r, L_t):
    clusters = initial_clusters.copy()
    
    # Create list of clusters that can still increase in size
    incomplete_clusters = [i for i, cluster in enumerate(clusters)]
    
    while len(incomplete_clusters) > 1:
        # Randomly choose one of the incomplete clusters
        cluster_index = random.choice(incomplete_clusters)
        cluster = clusters[cluster_index]
        
        # Find all possible merge candidates for this cluster
        merge_candidates = []
        for j in incomplete_clusters:
            if j != cluster_index:
                if (len(cluster.robots) + len(clusters[j].robots) <= L_r and 
                    len(cluster.tasks) + len(clusters[j].tasks) <= L_t):
                    merge_candidates.append(j)

        if not merge_candidates:
            # If there are no possible merge pairs, remove this cluster from incomplete_clusters
            incomplete_clusters.remove(cluster_index)
        else:
            # Calculate capability matching scores to merge candidates
            cap_match_scores = np.array([capability_match_filter(cluster, clusters[i]) for i in merge_candidates])
            
            #print(f"Capability match scores: {np.round(cap_match_scores, 2)}")

            # Calculate merge probabilities based on capability matching scores
            sum = np.sum(np.abs(cap_match_scores))
            if sum == 0:
                #merge_probs = np.ones(len(cap_match_scores)) / len(cap_match_scores)
                """Actually, we should just mark this cluster as complete and remove it from incomplete_clusters"""
                incomplete_clusters.remove(cluster_index)
                #print(f"Cluster {cluster_index} has no capability match with any other cluster. Marking as complete.")
            else:
                # Normalize the scores to get probabilities
                merge_probs = np.abs(cap_match_scores) / sum
            
                #print(f"Merge probabilities: {np.round(merge_probs,2)}")

                # Choose a merge candidate based on probabilities
                merge_index = np.random.choice(len(merge_candidates), p=merge_probs)
                merge_cluster_index = merge_candidates[merge_index]
            
                # Perform the merge
                cluster.merge(clusters[merge_cluster_index])
                clusters.pop(merge_cluster_index)
            
                # Remove the merged cluster from incomplete_clusters
                if merge_cluster_index in incomplete_clusters:
                    incomplete_clusters.remove(merge_cluster_index)
            
                # Adjust indices in incomplete_clusters
                incomplete_clusters = [i if i < merge_cluster_index else i-1 for i in incomplete_clusters]
    
    return clusters

def refine_clusters_spatial_merge(initial_clusters, L_r, L_t, temp):
    clusters = initial_clusters.copy()
    
    # Create list of clusters that can still increase in size
    incomplete_clusters = [i for i, cluster in enumerate(clusters)]
    
    while len(incomplete_clusters) > 1:
        # Randomly choose one of the incomplete clusters
        cluster_index = random.choice(incomplete_clusters)
        cluster = clusters[cluster_index]
        
        # Find all possible merge candidates for this cluster
        merge_candidates = []
        for j in incomplete_clusters:
            if j != cluster_index:
                if (len(cluster.robots) + len(clusters[j].robots) <= L_r and 
                    len(cluster.tasks) + len(clusters[j].tasks) <= L_t):
                    merge_candidates.append(j)

        if not merge_candidates:
            # If there are no possible merge pairs, remove this cluster from incomplete_clusters
            incomplete_clusters.remove(cluster_index)
        else:
            # Calculate distances to merge candidates
            distances = [euclidean(cluster.get_center_of_mass(), clusters[j].get_center_of_mass()) 
                         for j in merge_candidates]
            
            # Calculate merge probabilities based on distances
            # Temp ratio argument defines the ratio in probabilities between 0 and max possible distance
            # EX: if d = 0 and prob = 0.1, with a ratio of 10, then max_d will have prob = 0.01
            merge_probs = temp_softmax(distances, temp)
            
            # Choose a merge candidate based on probabilities
            merge_index = np.random.choice(len(merge_candidates), p=merge_probs)
            merge_cluster_index = merge_candidates[merge_index]
            
            # Perform the merge
            cluster.merge(clusters[merge_cluster_index])
            clusters.pop(merge_cluster_index)
            
            # Remove the merged cluster from incomplete_clusters
            if merge_cluster_index in incomplete_clusters:
                incomplete_clusters.remove(merge_cluster_index)
            
            # Adjust indices in incomplete_clusters
            incomplete_clusters = [i if i < merge_cluster_index else i-1 for i in incomplete_clusters]
    
    return clusters

def generate_clusters_heuristic(assignment, robot_list, task_list, temp, hypes):
    
    L_r = hypes['L_r']  # Maximum number of robots in a cluster
    L_t = hypes['L_t']  # Maximum number of tasks in a cluster
    kappa = hypes['kappa']  # Kappa value for Cluster class

    # Convert assignment to clusters
    initial_clusters = convert_assignment_to_clusters(assignment, robot_list, task_list, kappa)
    
        

    # Refine clusters using heuristic method
    refined_clusters = refine_clusters_heuristic(initial_clusters, L_r, L_t, temp)
    
    return refined_clusters

def generate_clusters_spatial(assignment, robot_list, task_list, L_r, hypes):
    
    L_t = hypes['L_t']  # Maximum number of tasks in a cluster
    temp = hypes['temp']  # Temperature for softmax function
    kappa = hypes['kappa']  # Kappa value for Cluster class

    # Convert assignment to clusters
    initial_clusters = convert_assignment_to_clusters(assignment, robot_list, task_list,kappa)
    
    # Refine clusters using heuristic method
    refined_clusters = refine_clusters_spatial_merge(initial_clusters, L_r, L_t, temp)
    
    return refined_clusters

def generate_clusters_cap_match(assignment, robot_list, task_list, L_r, hypes):
    
    L_t = hypes['L_t']  # Maximum number of tasks in a cluster
    temp = hypes['temp']  # Temperature for softmax function
    kappa = hypes['kappa']  # Kappa value for Cluster class

    # Convert assignment to clusters
    initial_clusters = convert_assignment_to_clusters(assignment, robot_list, task_list, kappa)
    
    # Refine clusters using heuristic method
    refined_clusters = refine_clusters_cap_match_filter(initial_clusters, L_r, L_t)
    
    return refined_clusters

def generate_clusters_rand(assignment, robot_list, task_list, hypes):
    
    L_t = hypes['L_t']  # Maximum number of tasks in a cluster
    L_r = hypes['L_r']  # Maximum number of robots in a cluster
    kappa = hypes['kappa']  # Kappa value for Cluster class

    # #Print Assignment
    # print("Current Assignment:")
    # for task_id, robot_ids in assignment.items():
    #     if task_id == -1:
    #         print(f"Unassigned Robots: {robot_ids}")
    #     else:
    #         print(f"Task {task_id} assigned Robots: {robot_ids}")

    # Convert assignment to clusters
    initial_clusters = convert_assignment_to_clusters(assignment, robot_list, task_list, kappa)
        
    # for cluster in initial_clusters:
    #     print(f"Initial cluster: ")
    #     for robot_id in cluster.robot_ids:
    #         print(f"Robot: {robot_id} ")
    #     for task_id in cluster.task_ids:
    #         print(f"Task: {task_id} ")
    
    # Refine clusters using random merge strategy
    refined_clusters = refine_clusters_random_merge(initial_clusters, L_r, L_t)

    # print(f"Refined clusters: {len(refined_clusters)}")
    # for cluster in refined_clusters:
    #     print(f"Cluster with {len(cluster.robot_ids)} robots and {len(cluster.task_ids)} tasks")
    #     print(f"Robot IDs: {cluster.robot_ids}")
    #     print(f"Task IDs: {cluster.task_ids}")
    
    return refined_clusters

