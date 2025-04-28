"""
Code for handling quaternions by making it rotate.

@Author Mathilde Borsodi Hegna
"""

import numpy as np


class Quaternion:
    scalar: int
    "Scalar for the quaternion"
    vector: np.ndarray
    "Vector array"

    def __init__(self, scalar: int, vector: np.ndarray) -> None:
        """
        Class to handle quaternions.

        :param scalar: Scalar for the quaternion.
        :param vector: Vector for the quaternion.
        """
        vector = np.array(vector).reshape(-1)
        if (len(vector) != 3) or not (isinstance(1.0 * scalar, float)):
            raise ValueError(
                "Cannot initialize quaternion with fewer or more than four parameters"
            )
        self.scalar = scalar
        self.vector = vector

    def __add__(self, other):
        """
        Addition of two quaternions.

        :return: A new quaternion.
        """
        return Quaternion(self.scalar + other.scalar, self.vector + other.vector)

    def __mul__(self, other):
        """
        Multiplication between two quaternions.

        :param other: Another quaternion.
        :return: A new quaternion.
        """
        s1, v1 = self.scalar, self.vector
        s2, v2 = other.scalar, other.vector
        scalar_part = s1 * s2 - np.dot(v1, v2)
        vector_part = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(scalar_part, vector_part)

    def __abs__(self):
        """
        Absolute value of a quaternion.

        :return: Float value
        """
        return np.sqrt(self.scalar**2 + np.dot(self.vector, self.vector))

    def inverse(self):
        """
        Inverse of a quaternion

        :return: A new quaternion
        """
        norm_squared = self.scalar**2 + np.dot(self.vector, self.vector)
        if norm_squared == 0:
            raise ValueError("Cannot invert a zero quaternion")
        return Quaternion(self.scalar / norm_squared, -self.vector / norm_squared)

    @classmethod
    def from_rotation(cls, vector: np.ndarray, angle: float):
        """
        Create a quaternion representing a rotation around an axis.

        :param vector: The axis of rotation as a 3D vector.
        :param angle: The angle of rotation in radians.
        :return: A new quaternion representing the specified rotation.
        """
        vector = np.array(vector).reshape(-1)
        vector = vector / np.linalg.norm(vector)
        return cls(np.cos(angle / 2), np.sin(angle / 2) * vector)

    def rotate_vectors(self, vectors: np.ndarray):
        """
        Rotates vector using a quaternion.

        :param: Array of 3D vectors to rotate.
        :return: Array of rotated vectors.
        """
        if vectors.ndim == 1:
            # Input is 1D, makes it 2D
            vector_count = len(vectors) // 3
            vectors = vectors.reshape(vector_count, 3)
        if vectors.shape[1] != 3:
            raise ValueError("Each column should be a 3D vector")

        inverse = self.inverse()
        rotated_vectors = []
        for i in range(len(vectors)):
            q_vector = Quaternion(0, vectors[i])
            rotated_q_vector = self * q_vector * inverse
            rotated_vectors.append(rotated_q_vector.vector)

        return np.array(rotated_vectors)
