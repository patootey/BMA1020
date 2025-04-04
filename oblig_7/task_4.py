import numpy as np
from scipy.spatial import Delaunay
from quaternions import Quaternion
np.random.seed(18140517) # We fixate the randomness; do not adapt this code(!)


class Volume:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.n = np.size(self.elements, axis=0)
        self.m = np.size(self.nodes, axis=0)
        self.volumes = np.zeros(self.n)# volumes of the elements of the triangulatioon
        # the code under is not provided (or only in skeleton)
        for i, el in enumerate(self.elements):
            # (0) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            # compute the volumes of the elements in the triangulation
            # volumes[i] = ...
        # (B) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #self.volume =  # volume of the entire structure
        # The following line just needs to be uncommented once the other ingredients work
        #self.barycentre = self.integrate_centroid(lambda x : x)/self.volume
        
    def integrate_centroid(self, f):
        # Integrate the function f over the domain self using function values at
        # the centroids of the elements
        integral = 0
        # (A) Complete the code ::::::::::::::::::::::::::::::::::::::::::::::::
        #
        #
        return integral
    
    def inertia_integrand(self, x):
        # (C) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # function in the definition of the intertia tensor
        return 
        
# Import a teapot
A = np.fromfile('teapot_nodes.txt', sep=' ')
A = np.reshape(A, (int(np.size(A)/3),3)) + [0,0,100]
E = np.fromfile('teapot_elements.txt', sep=' ', dtype=int)
E = np.reshape(E, (int(np.size(E)/4),4))

E = E[:,1:]
# shift it nearer its centre
middle = np.sum(A, axis=0)/np.size(A,axis=0)
A = A - middle
# add som random interior points
Q = np.random.rand(100,3)
points = np.vstack((A,Q))

# Delaunay triangulate the result
# This gives the convex hull of the teapot
# But at least it is a volume
tri = Delaunay(points)
V = Volume(tri.points, tri.simplices)
# (D) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Inertia_tensor = 0  