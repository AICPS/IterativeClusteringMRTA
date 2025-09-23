# IterativeClusteringMRTA
Simulation and Algorithm code to support the paper "Collaborative Task Allocation for Heterogeneous Multi-Robot Systems Through Iterative Clustering"


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
Example Output (Figure 2 in paper):

<img width="450" height="300" alt="average_utility_per_iteration" src="https://github.com/user-attachments/assets/dffa8eec-d455-49c2-93ed-f2485780dc95" />

To test cluster size effects on utility over time:

```bash
python tests/cluster_size_comparison_time.py
```
Example Output (Figure 2 in paper):

<img width="450" height="300" alt="average_utility_vs_time" src="https://github.com/user-attachments/assets/5255add3-87be-4fd6-bc52-bd5b4fb7daa0" />

## Iterative Clustering Compared to Optimal for Small Problems
For small-scale problems, the clustering methods can be directly compared against optimal solutions. The tests/iteration_optimal_compared.py script directly compares the solutions produced by iterative clustering against the optimal solution for 500 randomly generated problems with nu = 10 robots and mu = 5 tasks.

| Method | Exact Optimal Matches (%) | Average Utility Ratio (%) |
|:-----|:------:|------:|
| Random Iterative Clustering |   99.6    |     99.9 |
| Heuristic Iterative Clustering |   99.0    |     99.9 |


