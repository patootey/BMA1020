import numpy as np


class Surface:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.n = np.size(self.elements, axis=0)
        self.m = np.size(self.nodes, axis=0)
        self.normals = np.zeros((self.n, 3))
        self.areas = np.zeros(self.n)
        # find normals and areas (fill this in yourself!)
        for i, el in enumerate(self.elements):
            p0, p1, p2 = self.nodes[el]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            normalize = np.linalg.norm(normal)
            self.areas[i] = 0.5 * normalize
            self.normals[i] = normal / normalize  # unit normal

    def integrate_surface_centroid(self, f):
        # integrates vector field f over surface
        # requires f to vectorize by rows
        # i.e. f(A)[i] = f(A[i])
        integral = 0
        for i, el in enumerate(self.elements):
            centroid = np.mean(self.nodes[el], axis=0)
            flux_density = f(centroid)
            normal = self.normals[i]
            area = self.areas[i]
            integral += area * np.dot(flux_density, normal)
        return integral


A = 100 * np.transpose(
    np.array(
        [
            [-1, 1, -1, 1, -1, 1, -1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )
)

# a)

E = np.array(
    [
        [0, 1, 2],
        [1, 3, 2],
        [0, 4, 1],
        [1, 4, 5],
        [1, 5, 3],
        [3, 5, 7],
        [3, 7, 2],
        [2, 7, 6],
        [2, 6, 0],
        [0, 6, 4],
        [4, 6, 5],
        [5, 6, 7],
    ]
)

# The full room
Z = Surface(A, E)

# A chosen wall
Z2 = Surface(A, E[0:2])

# b)


def f(x):
    return np.array([0, 0, 5])


# This should give zero
Z.integrate_surface_centroid(f)

# But this will not
b = Z2.integrate_surface_centroid(f)

# c)

# load a finer mesh of the same room
# otherwise the integral will be inaccurate, and
# normalization by hand and by numerical integration will then be different

A = np.loadtxt("F_nodes.csv", delimiter=";")
B = np.loadtxt("F_elements.csv", delimiter=";", dtype=np.int32)

# the full room
F_s = Surface(A, B)

# the desired wall
F_s_2 = Surface(A, B[0:161])

source = np.array([50, 0, 0])


def g(x):
    r = x - source
    r_normalize = np.linalg.norm(r)
    if r_normalize == 0:
        return np.array([0.0, 0.0, 0.0])
    return (100 / (4 * np.pi * r_normalize**2)) * (r / r_normalize)


# This should give 100 (if you are unsure about the proportionality constant in g)
F_s.integrate_surface_centroid(g)

c = F_s_2.integrate_surface_centroid(g)


print(np.round(b))
