import numpy as np
from quaternions import Quaternion

np.random.seed(18140517)  # We use randomness in the test below. Do NOT change this line


def closest(s, t):
    # input: (ordered) list of (unique) time instances t
    # point where function to be evaluated s
    # output: index of sub interval
    if s < t[0] or s > t[-1]:
        return -1

    for i in range(len(t) - 1):
        if t[i] <= s < t[i + 1]:
            return i

    return -1  # default value


def interpolate_quaternion(Rotations, times, t):
    # find index corresponding time immediately before t
    # returns -1 if t is less than all times
    i = closest(t, times)
    if i == -1:
        return False
    else:
        t0 = times[i]
        t1 = times[i + 1]

        s = (t - t0) / (t1 - t0)

        q0_scalar = Rotations.scalar[i]
        q0_vec = Rotations.vec[:, i].reshape(3, 1)
        q1_scalar = Rotations.scalar[i + 1]
        q1_vec = Rotations.vec[:, i + 1].reshape(3, 1)

        q0 = Quaternion([q0_scalar], q0_vec)
        q1 = Quaternion([q1_scalar], q1_vec)

        result_scalar = (1 - s) * q0.scalar + s * q1.scalar
        result_vec = (1 - s) * q0.vec + s * q1.vec

        result = Quaternion(result_scalar, result_vec)

        norm = abs(result)
        result.scalar = result.scalar / norm
        result.vec = result.vec / norm

        return result


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

[[0.65106]
 [0.72991]
 [0.20472]]
[-0.03784]
"""
