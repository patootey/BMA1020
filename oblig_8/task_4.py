import numpy as np

# imports the function hermite(s,t,x,v)
# this functions as per exercise 2 from assignment 7
from hermite_util import hermite

# a)

# import a list of values of l(lambda), m(lambda), s(lambda)
# for values of lambda from 390 to 810 nanometres
# A[:,0] are the values of lambda
# A[:,1] the corresponding values of l(lambda)
# A[:,2] are m(lambda), and A[:,3] those of s(lambda)
A = np.loadtxt("LMS.csv", delimiter=";")

# First, interpolate to get values of L,M,S(lambda)
# Use Hermite interpolation
# hermite(s,t,x,v) interpolates data x(t), v(t) at point s


def hermite2(t, x):
    # returns a function f(t)
    # from data t, x
    # creates v by forward differences on t,x
    n = len(t)
    v = np.zeros(n)

    # Calculate derivatives using forward differences
    # i.e v[0] = (x[1]-x[0]) / (t[1]-t[0])
    for i in range(n - 1):
        v[i] = (x[i + 1] - x[i]) / (t[i + 1] - t[i])
    # the final element v[-1] 0 by the np.zeros initialization

    # Return a function that evaluates the interpolation at any point
    def interpolate(s):
        if s < t[0] or s > t[-1]:
            return 0
        pos, _ = hermite(s, t, x, v)
        return pos

    return interpolate


# the functions L, M, S
# normalized to max value 1
gl = hermite2(A[:, 0], A[:, 1])
gm = hermite2(A[:, 0], A[:, 2])
gs = hermite2(A[:, 0], A[:, 3])


# b)
def LMS(x):
    return np.array([gl(x), gm(x), gs(x)])


# Integrate so that E(lambda) gives an (L,M,S) value

x = np.linspace(390, 830, 500)


# Takes as argument a function E(x)
# returns the LMS values by integrating E*LMS with the trapezoid rule
# across the interval of interpolation with 500 points
# output: an array with 3 elements.
def Spectrum_to_LMS(E):
    # Create array of wavelengths

    # Evaluate E at each wavelength
    E_values = np.array([E(xi) for xi in x])

    # Calculate the L, M, S responses at each wavelength
    lms_values = np.array([LMS(xi) for xi in x])

    # Multiply E with each of L, M, S
    integrands = lms_values.T * E_values

    # Integrate using trapezoid rule
    dx = (x[-1] - x[0]) / (len(x) - 1)

    # Apply trapezoid rule
    results = np.zeros(3)
    for i in range(3):
        # End points with weight 0.5
        results[i] = 0.5 * dx * (integrands[i][0] + integrands[i][-1])
        # Inner points with weight 1.0
        results[i] += dx * np.sum(integrands[i][1:-1])

    return results


# c)


def R(x):
    return np.exp(-((x - 620) ** 2) / 1000) / np.sqrt(2000)


def G(x):
    return np.exp(-((x - 510) ** 2) / 1000) / np.sqrt(2000)


def B(x):
    return np.exp(-((x - 460) ** 2) / 1000) / np.sqrt(2000)

    # A program to convert (R,G,B) to (L,M,S)


R_LMS = Spectrum_to_LMS(R)  # integrate L2 against (gl, gm, gs)
G_LMS = Spectrum_to_LMS(G)  # ... M2 ...
B_LMS = Spectrum_to_LMS(B)  # ... S2 ...

# These can be assembled to give a matrix that converts from (R,G,B) to (L,M,S).

B = np.column_stack((R_LMS, G_LMS, B_LMS))

# for interest (not tested!), the RGB value of spectral yellow
# you should find that the B value is negative
np.linalg.solve(B, LMS(580))

print(np.round(Spectrum_to_LMS(lambda x: 0.4 * x**0.25)))
