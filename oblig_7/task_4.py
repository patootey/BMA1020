import numpy as np
from scipy.spatial import Delaunay
from quaternions import Quaternion

np.random.seed(18140517)  # We fixate the randomness; do not adapt this code(!)


class Volume:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.n = np.size(self.elements, axis=0)
        self.m = np.size(self.nodes, axis=0)
        self.volumes = np.zeros(self.n)  # volumes of the elements of the triangulation

        # (0) Compute the volumes of each element (tetrahedron)
        for i, el in enumerate(self.elements):
            v0 = self.nodes[el[0]]
            v1 = self.nodes[el[1]]
            v2 = self.nodes[el[2]]
            v3 = self.nodes[el[3]]

            a = v1 - v0
            b = v2 - v0
            c = v3 - v0

            self.volumes[i] = np.abs(np.dot(a, np.cross(b, c))) / 6.0

        # (B) Compute the total volume of the body
        self.volume = np.sum(self.volumes)

        # Compute the barycenter (center of mass) of the body
        self.barycentre = self.integrate_centroid(lambda x: x) / self.volume

    def integrate_centroid(self, f):
        # Integrate the function f over the domain self using function values at
        # the centroids of the elements
        integral = 0

        # (A) Integration over each element
        for i, el in enumerate(self.elements):
            v0 = self.nodes[el[0]]
            v1 = self.nodes[el[1]]
            v2 = self.nodes[el[2]]
            v3 = self.nodes[el[3]]

            centroid = (v0 + v1 + v2 + v3) / 4.0

            f_value = f(centroid)
            integral += f_value * self.volumes[i]

        return integral

    def inertia_integrand(self, x):
        # (C) Compute the integrand for the moment of inertia calculation
        r = x - self.barycentre

        r_squared_norm = np.sum(r**2)

        I3 = np.eye(3)

        r_outer_r = np.outer(r, r)

        return r_squared_norm * I3 - r_outer_r


# Import a teapot
A = np.fromfile("teapot_nodes.txt", sep=" ")
A = np.reshape(A, (int(np.size(A) / 3), 3)) + [0, 0, 100]
E = np.fromfile("teapot_elements.txt", sep=" ", dtype=int)
E = np.reshape(E, (int(np.size(E) / 4), 4))
E = E[:, 1:]
# shift it nearer its centre
middle = np.sum(A, axis=0) / np.size(A, axis=0)
A = A - middle
# add some random interior points
Q = np.random.rand(100, 3)
points = np.vstack((A, Q))
# Delaunay triangulate the result
# This gives the convex hull of the teapot
# But at least it is a volume
tri = Delaunay(points)
V = Volume(tri.points, tri.simplices)

# (D) Calculate the inertia tensor using the integrate_centroid method and the integrand
Inertia_tensor = V.integrate_centroid(V.inertia_integrand)
