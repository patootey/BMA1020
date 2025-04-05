import numpy as np
# imports the function hermite(s,t,x,v)
# this functions as per exercise 2 from assignment 7
from hermite_util import hermite

### a)

# import a list of values of l(lambda), m(lambda), s(lambda)
# for values of lambda from 390 to 810 nanometres
# A[:,0] are the values of lambda
# A[:,1] the corresponding values of l(lambda)
# A[:,2] are m(lambda), and A[:,3] those of s(lambda) 
A = np.loadtxt('LMS.csv', delimiter=';')

# First, interpolate to get values of L,M,S(lambda)
# Use Hermite interpolation
# hermite(s,t,x,v) interpolates data x(t), v(t) at point s
    
def hermite2(t,x):
    # returns a function f(t)
    # from data t, x 
    # creates v by forward differences on t,x
    # i.e v[0] = (x[1]-x[0]) / (t[1]-t[0])
    # the final element v[-1] = 0

# the functions L, M, S
# normalized to max value 1
gl = hermite2(A[:,0],A[:,1])
gm = hermite2(A[:,0],A[:,2])
gs = hermite2(A[:,0],A[:,3])

### b)
def LMS(x):
    return np.array([gl(x), gm(x), gs(x)])

# Integrate so that E(lambda) gives an (L,M,S) value 

x = np.linspace(390,830,500)

# Takes as argument a function E(x)
# returns the LMS values by integrating E*LMS with the trapezoid rule
# across the interval of interpolation with 500 points
# output: an array with 3 elements.
def Spectrum_to_LMS(E)
    return ...

### c) 

def R(x):
    return np.exp(-(x - 620)**2/1000)/np.sqrt(2000)

def G(x):
    return np.exp(-(x - 510)**2/1000)/np.sqrt(2000)

def B(x):
    return np.exp(-(x - 460)**2/1000)/np.sqrt(2000)
    
    # A program to convert (R,G,B) to (L,M,S)

R_LMS = # integrate L2 against (gl, gm, gs)
G_LMS = # ... M2 ...
B_LMS = # ... S2 ...

# These can be assembled to give a matrix that converts from (R,G,B) to (L,M,S).

B = ...

# for interest (not tested!), the RGB value of spectral yellow
# you should find that the B value is negative
np.linalg.solve(B,LMS(580))