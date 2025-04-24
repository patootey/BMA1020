"""
Code for handling quaternions by making it rotate.

Author: Mathilde Borsodi Hegna
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

    # 10 points
    @classmethod
    def model_rect(
        cls,
        x_min: float,
        x_max: float,
        num_x_divisions: int,
        y_min: float,
        y_max: float,
        num_y_divisions: int,
        emission_color: np.ndarray = np.array([0, 0, 0]),
        reflect: np.ndarray = np.array([0.5, 0.5, 0.5]),
        translated_vector: np.ndarray = np.array([0, 0, 0]),
        rotation_axis: np.ndarray = np.array([0, 0, 1]),
        rotation_angle: float = 0,
    ):
        # divides a rectangle [x0,x1] x [y0,y1] located at z = 0 into an m x n grid
        # triangulates the grid in a regular manner
        # then translates and rotates the rectangle to the desired 3d position
        # the values of emit and reflect can be chosen
        x = np.linspace(x_min, x_max, num_x_divisions)
        y = np.linspace(y_min, y_max, num_y_divisions)
        grid_x, grid_y = np.meshgrid(x, y)
        Z = np.transpose(
            np.vstack(
                (
                    grid_x.reshape(num_y_divisions * num_x_divisions),
                    grid_y.reshape(num_y_divisions * num_x_divisions),
                )
            )
        )
        divisions = np.zeros((num_x_divisions * num_y_divisions, 1))
        vertices = np.hstack((Z, divisions))

        # Create triangles by connecting grid points
        triangle_indices = []

        # For an m×n grid of points, we create 2×(m-1)×(n-1) triangles
        for y_div in range(num_y_divisions - 1):
            for x_div in range(num_x_divisions - 1):
                # Calculate vertex indices for the current grid cell
                # The grid is laid out row by row
                top_left = y_div * num_x_divisions + x_div
                top_right = top_left + 1
                bottom_left = (y_div + 1) * num_x_divisions + x_div
                bottom_right = bottom_left + 1

                # Create two triangles per grid cell
                # Triangle 1: top_left, top_right, bottom_left (clockwise)
                # Triangle 2: bottom_left, top_right, bottom_right (clockwise)
                # This ensures consistent clockwise ordering for normal calculation
                triangle_indices.append([top_left, top_right, bottom_left])
                triangle_indices.append([bottom_left, top_right, bottom_right])

        triangle_indices = np.array(triangle_indices)

        # Create a quaternion for the rotation
        quaternion_rotation = cls.from_rotation(np.array(rotation_axis), rotation_angle)

        # Apply rotation and translation
        for y_div in range(len(vertices)):
            # First rotate using the quaternion
            quaternion_vertex = cls(0, vertices[y_div])

            # KAN ENDRE TIL ROATATE VECTORS I STEDENFOR
            rotated_vertex = (
                quaternion_rotation * quaternion_vertex * quaternion_rotation.inverse()
            )
            vertices[y_div] = rotated_vertex.vector

            # Then translate
            vertices[y_div] += np.array(translated_vector)

        # Create arrays for emission and reflection values for each triangle
        num_triangles = len(triangle_indices)
        emit_array = np.array([emission_color for _ in range(num_triangles)])
        reflect_array = np.array([reflect for _ in range(num_triangles)])

        # Return all components needed to create the rectangle model
        return vertices, triangle_indices, emit_array, reflect_array

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

        rotated_vectors = []
        for i in range(len(vectors)):
            q_vector = Quaternion(0, vectors[i])
            rotated_q_vector = self * q_vector * self.inverse()
            rotated_vectors.append(rotated_q_vector.vector)

        return np.array(rotated_vectors)
