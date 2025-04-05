import numpy as np

class Surface:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.n = np.size(self.elements, axis=0)
        self.m = np.size(self.nodes, axis=0)
        self.normals = np.zeros((self.n,3))
        self.areas = np.zeros(self.n)
        # find normals and areas (fill this in yourself!)
    
    def integrate_surface_centroid(self, f):
        # integrates vector field f over surface
        # requires f to vectorize by rows
        # i.e. f(A)[i] = f(A[i])
        integral = 0
        for i, el in enumerate(self.elements):
            # fill in
        return integral
        
A = 100*np.transpose(np.array([[-1,1,-1,1,-1,1,-1,1],[-1,-1,1,1,-1,-1,1,1],
                               [-1,-1,-1,-1,1,1,1,1]]))

### a)

E = # fill this in yourself!

# The full room
Z = Surface(A,E)

# A chosen wall
Z2 = Surface(A,E[0:2])

### b) 

def f(x):
    return # fill this in

# This should give zero
Z.integrate_surface_centroid(f)

# But this will not
b = Z2.integrate_surface_centroid(f)

### c)

# load a finer mesh of the same room
# otherwise the integral will be inaccurate, and
# normalization by hand and by numerical integration will then be different

A = np.loadtxt('F_nodes.csv',delimiter=';')
B = np.loadtxt('F_elements.csv', delimiter=';', dtype=np.int32)

# the full room
F_s = Surface(A,B)

# the desired wall
F_s_2 = Surface(A,B[0:161])

def g(x):
    return # fill this in

# This should give 100 (if you are unsure about the proportionality constant in g)
F_s.integrate_surface_centroid(g)

c = F_s_2.integrate_surface_centroid(g)