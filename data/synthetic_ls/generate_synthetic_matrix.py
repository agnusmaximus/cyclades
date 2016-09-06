from __future__ import print_function
import sys
import random

output_file = sys.argv[1]
n = int(sys.argv[2])
sparsity_percentage = float(sys.argv[3])
f_out = open(output_file, "w")

matrix = [[0 for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        if random.uniform(0, 1) < sparsity_percentage:
            matrix[i][j] = random.uniform(1, 1000)

print("%d" % n, file=f_out)
for i in range(n):
    row = matrix[i]
    rest_str = " ".join([str(ind) + " " + str(v) for ind,v in enumerate(row) if v != 0])
    line_str = "%d %s" % (i, rest_str)
    print(line_str, file=f_out)
f_out.close()
