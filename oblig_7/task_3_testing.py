import numpy as np
from quaternions import Quaternion

np.random.seed(18140517)  # We use randomness in the test below. Do NOT change this line


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


def interpolate_quaternion(Rotations, times, t):
    # find index corresponding time immediately before t
    # returns -1 if t is less than all times
    i = closest(t, times)
    if i == -1:
        return False
    else:
        # If t is exactly the last time value
        if i == len(times) - 1:
            return Quaternion(Rotations.scalar[i : i + 1], Rotations.vec[:, i : i + 1])

        # Calculate the interpolation parameter (0 to 1)
        t1 = times[i]
        t2 = times[i + 1]
        s = (t - t1) / (t2 - t1)  # Normalized time parameter

        # Linear interpolation between quaternions
        result_scalar = Rotations.scalar[i] * (1 - s) + Rotations.scalar[i + 1] * s
        result_vec = Rotations.vec[:, i] * (1 - s) + Rotations.vec[:, i + 1] * s

        # Convert to numpy arrays with proper shape
        scalar_array = np.array([result_scalar])
        vec_array = result_vec.reshape(3, 1)

        # Create the result quaternion
        result = Quaternion(scalar_array, vec_array)

        # Normalize to ensure it's a unit quaternion
        norm = abs(result)
        normalized_scalar = result.scalar / norm
        normalized_vec = result.vec / norm

        return Quaternion(normalized_scalar, normalized_vec)


# No need to change anything after this line :::::::::::::::::::::::::::::::::::
N = 3
tspan = np.linspace(0.0, 5, N)
vvec = np.random.rand(3, N)
sscalar = np.random.randn(N)
Rotations = Quaternion(sscalar, vvec)
s = 3
qnew = interpolate_quaternion(Rotations, tspan, s)


print(np.round(qnew.vec, 5))
print(np.round(qnew.scalar, 5))


"""
[[-0.39946]
 [-0.40014]
 [ 0.10287]]
[-0.81838]
"""
