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


def vector_normalize(vector: np.ndarray):
    """
    Normalizing the vector.
    """
    length = np.linalg.norm(vector)
    return vector / length


def array_vector_normalize(vectors: np.ndarray):
    """
    Normalizing the vector.
    """
    length = np.linalg.norm(vectors, axis=1)
    return vectors / length


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
    return np.dot(plane_n, point) - np.dot(plane_n, plane_p)


def array_distance_point_to_plane(
    plane_n: np.ndarray, plane_p: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Getting the distance between a point and plane.

    :param plane_n: The normal of the plane.
    :param plane_p: The position of the plane.
    :param point: The point position.
    """

    # Plane needs to be normalized
    plane_n = vector_normalize(plane_n)

    # Return shortes distance from point to plane
    return np.vecdot(plane_n, points) - np.vecdot(plane_n, plane_p)


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
    plane_d = -np.dot(plane_n, plane_p)
    ad = np.dot(point_1, plane_n)
    bd = np.dot(point_2, plane_n)

    # If line is parallel with the plane
    if ad == bd:
        return None

    t = (-plane_d - ad) / (bd - ad)

    vec = point_2 - point_1
    intersect = vec * t

    return point_1 + intersect
