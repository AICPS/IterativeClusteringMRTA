import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phase2.phase2_utils as utils
print('here')
n = 10
m = 10
L = 3

result = utils.generate_partitions(n,m,L)

print(f"All partitions of {n} into {m+1} parts with the initial team representing unassigned robots")
for partition in result:
    print(partition)

print(f"total number of partitions: {len(result)}")