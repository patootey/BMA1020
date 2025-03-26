import numpy as np


class Quaternion:
    def __init__(self, scalar: int, vec: np.ndarray):
        vec = np.array(vec).reshape(-1)
        if (len(vec) != 3) or not (isinstance(1.0 * scalar, float)):
            raise ValueError(
                "Cannot initialize quaternion with fewer or more than four parameters"
            )
        self.scalar = scalar

        self.vec = vec

    # Make sure the vertices can be modifided without affecting other triangles
    def __repr__(self):
        return f"quaternion {self.scalar} + i*{self.vec[0]} + j*{self.vec[1]} + k*{self.vec[2]}"

    def __add__(self, other):

        return Quaternion(self.scalar + other.scalar, self.vec + other.vec)

    def __mul__(self, other):

        s1, v1 = self.scalar, self.vec
        s2, v2 = other.scalar, other.vec
        scalar_part = s1 * s2 - np.dot(v1, v2)
        vector_part = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(scalar_part, vector_part)

    def __abs__(self):
        return np.sqrt(self.scalar**2 + np.dot(self.vec, self.vec))

    def inverse(self):
        norm_squared = self.scalar**2 + np.dot(self.vec, self.vec)
        if norm_squared == 0:
            raise ValueError("Cannot invert a zero quaternion")
        return Quaternion(self.scalar / norm_squared, -self.vec / norm_squared)

    @classmethod
    def from_rotation(cls, vector: np.ndarray, angle: float):
        vector = np.array(vector).reshape(-1)
        vector = vector / np.linalg.norm(vector)
        return cls(np.cos(angle / 2), np.sin(angle / 2) * vector)

    def rotate_vectors(self, vectors: np.ndarray):
        if vectors.ndim == 1:
            # Input is 1D, makes it 2D
            vector_count = len(vectors) // 3
            vectors = vectors.reshape(vector_count, 3)
        if vectors.shape[1] != 3:
            raise ValueError("Each column should be a 3D vector")

        rotated_vectors = []
        for i in range(len(vectors)):
            q_vector = Quaternion(0, vectors[i])
            rotated_q_vector = self * q_vector * self.inverse()
            rotated_vectors.append(rotated_q_vector.vec)

        return np.array(rotated_vectors)
