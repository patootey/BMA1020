import numpy as np


class Quaternion:
    def __init__(self, scalar, vec):
        if (len(vec) != 3) or not (isinstance(1.0 * scalar, float)):
            raise ValueError(
                "Cannot initialize quaternion with fewer or more than four parameters"
            )
        self.scalar = scalar
        self.vec = np.array(vec)

    def __repr__(self):
        return f"quaternion {self.scalar} + i*{self.vec[0]} + j*{self.vec[1]} + k*{self.vec[2]}"

    def __add__(self, other):
        return Quaternion(self.scalar + other.scalar, self.vec + other.vec)

    def __mul__(self, other):
        s1, v1 = self.scalar, self.vec
        s2, v2 = other.scalar, other.vec
        new_scalar = s1 * s2 - np.dot(v1, v2)
        new_vec = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(new_scalar, new_vec)

    def __abs__(self):
        return np.sqrt(self.scalar**2 + np.dot(self.vec, self.vec))

    def inverse(self):
        norm_sq = abs(self) ** 2
        return Quaternion(self.scalar / norm_sq, -self.vec / norm_sq)

    @classmethod
    def from_rotation(cls, vector, angle):
        return cls(np.cos(angle / 2), np.sin(angle / 2) * np.array(vector))

    def rotate_vector(self, vector):
        vector = np.asarray(vector).reshape(3, 1)
        vector_q = Quaternion(0, vector.flatten())
        rotated_q = self * vector_q * self.inverse()
        return rotated_q.vec.reshape(3, 1)


q1 = Quaternion.from_rotation(np.array([0, 1, 0]), np.pi / 2)
v_out = q1.rotate_vector(np.array([[3], [4], [-5]]))
print(np.round(v_out, 5))
