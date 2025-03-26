import numpy as np

# We create a random adjacency matrix for you to work with - don't change this code
np.random.seed(0)
M = np.random.rand(30, 30)  # a random matrix, entries are floats between 0..1
M[M > 0.9] = 1.0  # all entries > 0.9 become 1.0 and give arrows
M[M <= 0.9] = 0.0  # all entries <= 0.9 become 0.0, and give no arrows.

# a)
# Create the vector av where av[0]=1 and av[i]=0 for other i.
av = np.zeros(30)
av[0] = 1.0

# Compute the number of arrows from vertex 0 to vertex 5
a5 = int((M @ av)[5])

# Compute the number of arrows from vertex 0 to vertex 11
a11 = int((M @ av)[11])

# Compute the number of loops at vertex 0
numloops = int(M[0, 0])

# b)
# Compute the number of paths from vertex 0 to vertex 8 of length 2
num082 = int((M @ (M @ av))[8])

# Compute the number of paths from vertex 0 to vertex 10 of length 3
num0103 = int((M @ M @ (M @ av))[10])

# Compute the number of paths from vertex 0 to vertex 11 of length 5
num0115 = int((M @ M @ M @ M @ (M @ av))[11])
