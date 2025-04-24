"""
Code for handling vector functions and intersections & get the dataclass Trigon

Author: Mathilde Borsodi Hegna
"""

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


# Trigon class to shorten the code
@dataclass
class Trigon:
    vertex_1: np.ndarray
    "First vertex in trigon"
    vertex_2: np.ndarray
    "Second vertex in trigon"
    vertex_3: np.ndarray
    "Third vertex in trigon"
    index: int
    "Original index in triangle array"
    color: NDArray[np.int64]
    "Color of trigon"


def LMS_to_RGB(lms: np.ndarray):  # 5 points
    # Bradford's spectrally sharpened matrix (LLAB, CIECAM97s)
    conversion_matrix = np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ]
    )

    # Convert LMS to RGB
    rgb = np.dot(conversion_matrix, lms)

    # Ensure values are in valid range
    return np.clip(rgb, 0, 1)


def find_colors(self, n):  # 10 points
    pass


def find_luminances(self, n):  # 20 points
    pass


# General functions for vectors
def vector_dot_product(vector1: np.ndarray, vector2: np.ndarray):
    """
    Get the dot product of the vectors.
    """
    # Numpy does per element multiplication
    return np.sum(vector1 * vector2)


def vector_length(vector: np.ndarray):
    """
    Calculate length of the vector and return it.
    """
    return np.sqrt(vector_dot_product(vector, vector))


def vector_normalize(vector: np.ndarray):
    """
    Normalizing the vector.
    """
    length = vector_length(vector)
    return vector / length


def distance_point_to_plane(
    plane_n: np.ndarray, plane_p: np.ndarray, point: np.ndarray
) -> float:
    """
    Getting the distance between a point and plane.

    :param plane_n: The normal of the plane.
    :param plane_p: The position of the plane.
    :param point: The point position.
    """

    # Plane needs to be normalized
    plane_n = vector_normalize(plane_n)

    # Return shortes distance from point to plane
    return vector_dot_product(plane_n, point) - vector_dot_product(plane_n, plane_p)


def line_intersect_plane(
    plane_p: np.ndarray,
    plane_n: np.ndarray,
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> np.ndarray | None:
    """
    Finds the point where the line and plane intersects.
    The line is defined by two points.

    :param plane_n: The normal of the plane.
    :param plane_p: The position of the plane.
    :param point_1: First point on the line.
    :param point_2: Second point on the line.
    """
    # Found this on the net, intersect plane with line
    # Combined with the Youtube videos I have watched
    # Make sure the plane is normalized
    plane_n = vector_normalize(plane_n)

    # D element of the plane
    plane_d = -vector_dot_product(plane_n, plane_p)
    ad = vector_dot_product(point_1, plane_n)
    bd = vector_dot_product(point_2, plane_n)

    # If line is parallel with the plane
    if ad == bd:
        return None

    t = (-plane_d - ad) / (bd - ad)

    vec = point_2 - point_1
    intersect = vec * t

    return point_1 + intersect
