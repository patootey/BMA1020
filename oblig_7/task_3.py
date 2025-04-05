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
        # Adapt the code that follows within this function :::::::::::::::::::::
        return


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
