import numpy as np


# This code computes the form factor F_ij for elements i and j
# this returns a scalar number
def form_factor(nodes, elements, normals, areas, i, j):
    if i == j:
        return 0.0

    # Get the coordinates of the nodes for elements i and j

    centroid_i = np.mean(nodes[elements[i]], axis=0)
    centroid_j = np.mean(nodes[elements[j]], axis=0)

    # Vector from i to j
    r_vector = centroid_j - centroid_i
    r_normalize = np.linalg.norm(r_vector)

    if r_normalize == 0:
        return 0.0  # Avoid division by zero

    # Unit direction vector
    r_unit = r_vector / r_normalize

    dot_ni = np.dot(r_unit, normals[i])
    dot_nj = np.dot(-r_unit, normals[j])

    if dot_ni <= 0 or dot_nj <= 0:
        return 0.0  # Not visible

    # opphøyd i 2 fordi det er mindre triangel
    F_ij = areas[j] * (dot_ni * dot_nj) / (np.pi * r_normalize**2)

    return F_ij


# This code to assemble a form factor matrix for the given surface
def form_factor_matrix(nodes, elements, normals, areas):
    n = len(elements)
    F = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            F[i, j] = form_factor(nodes, elements, normals, areas, i, j)

    return F


# Load in an icosahedron/D20
# the format is familiar, N are the nodes, E the surface elements
# the normals and areas of each element are computed for you:
# Normals[i] is the normal vector of element i, etc.

N = np.loadtxt("d20_nodes.csv", delimiter=";")
E = np.loadtxt("d20_elements.csv", delimiter=";", dtype=np.int32)
Normals = np.loadtxt("d20_normals.csv", delimiter=";")
Areas = np.loadtxt("d20_areas.csv", delimiter=";")

A = form_factor_matrix(N, E, Normals, Areas)


print(np.size(A, axis=0))
print(np.size(A, axis=1))
print(np.round(A[10, 12], decimals=5))
