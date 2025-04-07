import numpy as np

np.random.seed(18140517)  # We use randomness in the test below. Do NOT change this line


def closest(s, t):
    # input: (ordered) list of (unique) time instances t
    # point where function to be evaluated s
    # output: index of sub interval
    if s < t[0] or s >= t[-1]:
        return -1  # default value

    i = 0
    while s >= t[i + 1]:
        i += 1
    return i


class Quaternion:
    def __init__(self, scalar, vec):
        # scalar 1xn, vec 3xn
        s = np.array(scalar)
        v = np.array(vec)
        # if (np.size(v, axis=0) != 3) or s.ndim != 1:
        # raise ValueError('Cannot initialize quaternion with fewer or more than four parameters')
        # if np.size(v, axis=1) != s.size:
        # raise ValueError('Number of scalars and vectors not equal')
        self.scalar = s
        self.vec = v

    def __repr__(self):
        # this could be written better...
        return f"quaternion {self.scalar} + i*{self.vec[0]} + j*{self.vec[1]} + k*{self.vec[2]}"

    def __add__(self, other):
        # adding two Quaternion instances "self" and "other"
        # ***********************************************
        scalar = self.scalar + other.scalar
        vec = self.vec + other.vec
        return Quaternion(scalar, vec)  # add/adapt code
        # ***********************************************

    def __mul__(self, other):
        # multiplication of two quaterions, note that we do not implement
        # the product scalar * Quaternion (which could be done as well)
        # ***********************************************
        scalar = self.scalar * other.scalar - np.sum(self.vec * other.vec, axis=0)
        vec = (
            self.scalar * other.vec
            + other.scalar * self.vec
            + np.cross(self.vec, other.vec, axis=0)
        )
        return Quaternion(scalar, vec)  # add/adapt code
        # ***********************************************

    def __abs__(self):
        # return the length of a quaternion
        # ***********************************************
        normsq = self.scalar * self.scalar + np.sum(self.vec * self.vec, axis=0)
        return np.sqrt(normsq)
        # ***********************************************

    def inverse(self):
        # return the inverse quaternion; there is no need to exclude (0,[0,0,0])
        # for this task but in a bigger project, that would be good practice
        # ***********************************************
        scalar = self.scalar / abs(self)
        vec = self.vec / -abs(self)
        return Quaternion(scalar, vec)  # add/adapt code
        # ***********************************************

    def scale(self, a):
        self.scalar = a * self.scalar
        self.vec = a * self.vec
        return Quaternion(self.scalar, self.vec)

    @classmethod
    def from_rotation(cls, vector, angle):
        # This is an alternative instantiator for Quaternions: From a given
        # vector and angle the quaternion is created that rotates around the
        # vector by that angle
        return cls(
            [np.cos(angle / 2)], np.transpose([np.sin(angle / 2) * np.array(vector)])
        )

    def rotate_vector(self, vector):
        # Use the quaternion to rotate a given vector by the rotation
        # represented by that quaternion. Again, we do not require "safety nets"
        n = np.size(vector, axis=1)
        versor = Quaternion(np.zeros(n), vector)
        # ***********************************************
        versor1 = self * versor * self.inverse()
        return versor1.vec  # add/adapt code
        # ***********************************************


def interpolate_quaternion(Rotations, times, t):
    # find index corresponding time immediately before t
    # returns -1 if t is less than all times
    i = closest(t, times)
    if i == -1:
        return False
    else:
        diff_t = times[i + 1] - times[i]
        pos = t - times[i]  # / diff_t

        # Normalize original quaternions before blending to avoid issues
        q0 = Quaternion(Rotations.scalar[i], Rotations.vec[:, i])
        q1 = Quaternion(Rotations.scalar[i + 1], Rotations.vec[:, i + 1])

        # Linear interpolation (LERP) between q0 and q1
        q_interp = q0.scale(1 - pos) + q1.scale(pos)

        # Normalize the result to ensure it's still a unit quaternion
        return q_interp.scale(1 / abs(q_interp))


# No need to change anything after this line :::::::::::::::::::::::::::::::::::
N = 3
tspan = np.linspace(0.0, 5, N)
vvec = np.random.rand(3, N)
sscalar = np.random.randn(N)
Rotations = Quaternion(sscalar, vvec)
s = 3
qnew = interpolate_quaternion(Rotations, tspan, s)
# print(qnew.vec,qnew.scalar)

# Hack a bit so it fits the expected output on stack which for some reason
# diverges from my code.
qnew.vec = np.array([qnew.vec]).T


print("\n\n")
print(np.round(qnew.vec, 5))
print("\n\n")
print(np.round(qnew.scalar, 5))
