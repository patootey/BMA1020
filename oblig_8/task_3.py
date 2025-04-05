# This code computes the form factor F_ij for elements i and j
# this returns a scalar number
def form_factor(nodes, elements, normals, areas, i, j):
    return  # complete


# This code to assemble a form factor matrix for the given surface
def form_factor_matrix(nodes, elements, normals, areas):
    return  # complete


# Load in an icosahedron/D20
# the format is familiar, N are the nodes, E the surface elements
# the normals and areas of each element are computed for you:
# Normals[i] is the normal vector of element i, etc.

N = np.loadtxt("d20_nodes.csv", delimiter=";")
E = np.loadtxt("d20_elements.csv", delimiter=";", dtype=np.int32)
Normals = np.loadtxt("d20_normals.csv", delimiter=";")
Areas = np.loadtxt("d20_areas.csv", delimiter=";")

A = form_factor_matrix(N, E, Normals, Areas)
