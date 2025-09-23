# IterativeClusteringMRTA
This repositoy contains source code to support our paper "Collaborative Task Allocation for Heterogeneous Multi-Robot Systems Through Iterative Clustering".

## Abstract
Multi-robot systems face the challenge of efficiently allocating teams of heterogeneous robots to tasks. The task allocation problem is complicated by collaborative interactions between robots where teams of robots developemergent capabilities that enable them to complete tasks that would be inefficient or impossible for individual robots. To address these challenges, we present an iterative clustering algorithm for collaborative task allocation in heterogeneous multi-robot systems. This approach partitions the computationally intractable global optimization problem into smaller, tractable subproblems by iteratively forming clusters of robots and tasks, then optimizing assignments within each cluster. By ensuring robots remain clustered with their currently assigned tasks, we guarantee monotonic improvement in overall utility with each iteration. We analyze the convergence of the algorithm and characterize how cluster size constraints determine which suboptimal assignments could trap the algorithm. In simulation, iterative clustering consistently outperforms simulated annealing, and a group-based auction in both computation time and solution quality, and outperforms a hedonic game approach in solution quality.

## Setup
Create the conda environment:

```bash
conda env create -f environment.yml
conda activate icmrta
```

## Cluster Size Comparison
The below scripts test how cluster size limits affect the performance of the algorithm.

To test cluster size effects on utility per iteration:
```bash
python tests/cluster_size_comparison_iterations.py
```
### Results:

<img width="400" height="267" alt="average_utility_per_iteration" src="https://github.com/user-attachments/assets/dffa8eec-d455-49c2-93ed-f2485780dc95" />

To test cluster size effects on utility over time:

```bash
python tests/cluster_size_comparison_time.py
```
### Results:

<img width="400" height="267" alt="average_utility_vs_time" src="https://github.com/user-attachments/assets/5255add3-87be-4fd6-bc52-bd5b4fb7daa0" />

## Iterative Clustering Compared to Optimal for Small Problems
For small-scale problems, the clustering methods can be directly compared against optimal solutions. The tests/iteration_optimal_compared.py script directly compares the solutions produced by iterative clustering against the optimal solution for 500 randomly generated problems with nu = 10 robots and mu = 5 tasks.

```bash
python tests/iteration_optimal_compared.py
```

### Results:
| Method | Exact Optimal Matches (%) | Average Utility Ratio (%) |
|:-----|:------:|------:|
| Random Iterative Clustering |   99.6    |     99.9 |
| Heuristic Iterative Clustering |   99.0    |     99.9 |

## Comparison to Representative Task Allocation Methods.
To evaluate the iterative clustering algorithm's effectiveness on a larger problem with nu = 50 robots and mu = 25 tasks, the random and heuristic iterative clustering algorithms are compared against group-based auction, hedonic game,
and simulated annealing approaches. Overall, heuristic clustering outperformed the other methods, closely followed by random clustering.

```bash
python tests/method_time_comparison.py
```

### Results:

<img width="375" height="300" alt="5_method_time_comparison" src="https://github.com/user-attachments/assets/2026e497-12ab-4b71-98a4-5e73c25b0241" />

## Scalability Testing:
The scalability of both the random and heuristic iterative clustering algorithms is tested on four problem sizes with results averaged over 100 trials. Random clustering scaled more efficiently than heuristic clustering due to the computational cost of calculating the heuristics for cluster formation.

```bash
python tests/scalability_test.py
```

### Results:

<img width="375" height="307" alt="new_scalability_new_size" src="https://github.com/user-attachments/assets/d1996f27-14f0-4a23-98f9-90f65e9ec298" />


