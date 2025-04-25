import numpy as np


def closest(s, t):
    # input: (ordered) list of (unique) time instances t
    # point where function to be evaluated s
    # output: index of sub interval

    # Check if s is within the range of t
    if s < t[0] or s > t[-1]:
        return -1

    # Find the index i such that t[i] <= s < t[i+1]
    for i in range(len(t) - 1):
        if t[i] <= s < t[i + 1]:
            return i

    return -1  # default value if no interval is found


def hermite(s, t, x, v):
    # import list of time steps t, list of positions x
    # and list of speeds v
    # returns x(s), v(s) by Hermite interpolation
    i = closest(s, t)
    if i == -1:
        return 0, 0
    else:
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        v0, v1 = v[i], v[i + 1]

        # Normalize the parameter
        h = t1 - t0
        u = (s - t0) / h

        # Cubic Hermite basis functions
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2

        pos = h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1

        dh00 = (6 * u**2 - 6 * u) / h
        dh10 = 3 * u**2 - 4 * u + 1
        dh01 = (-6 * u**2 + 6 * u) / h
        dh11 = 3 * u**2 - 2 * u

        vel = dh00 * x0 + dh10 * v0 + dh01 * x1 + dh11 * v1

        return pos, vel


LMS_matrix = np.loadtxt("LMS.csv", delimiter=";")


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
gl = hermite2(LMS_matrix[:, 0], LMS_matrix[:, 1])
gm = hermite2(LMS_matrix[:, 0], LMS_matrix[:, 2])
gs = hermite2(LMS_matrix[:, 0], LMS_matrix[:, 3])


# b)
def LMS(x):
    return np.array([gl(x), gm(x), gs(x)])


# Integrate so that E(lambda) gives an (L,M,S) value
x = np.linspace(390, 830, 500)


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


def Red(x):
    return np.exp(-((x - 620) ** 2) / 1000) / np.sqrt(2000)


def Green(x):
    return np.exp(-((x - 510) ** 2) / 1000) / np.sqrt(2000)


def Blue(x):
    return np.exp(-((x - 460) ** 2) / 1000) / np.sqrt(2000)

    # A program to convert (R,G,B) to (L,M,S)


R_LMS = Spectrum_to_LMS(Red)  # integrate L2 against (gl, gm, gs)
G_LMS = Spectrum_to_LMS(Green)  # ... M2 ...
B_LMS = Spectrum_to_LMS(Blue)  # ... S2 ...

# These can be assembled to give a matrix that converts from (R,G,B) to (L,M,S).

Colors = np.column_stack((R_LMS, G_LMS, B_LMS))

# for interest (not tested!), the RGB value of spectral yellow
# you should find that the B value is negative
np.linalg.solve(Colors, LMS(580))


def LMS_to_RGB(luminances: np.ndarray):  # 5 points
    # B is the matrix that converts RGB to LMS
    # To convert from LMS to RGB, we need the inverse of B
    B_inverse = np.linalg.inv(Colors)

    # Convert LMS to RGB
    rgb_linear = np.dot(B_inverse, luminances)

    # Clip values to ensure they're within a valid range (0-1)
    rgb_linear = np.clip(rgb_linear, 0, 1)

    # From  https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    # gammma = 1 / 2.2

    # rgb_gamma_corrected = np.power(rgb_linear, gammma)

    rgb_255 = (rgb_linear * 255).astype(int)

    return rgb_255
