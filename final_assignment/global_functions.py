"""
Code for handling vector functions and intersections, raytracing

Author: Mathilde Borsodi Hegna
"""

import numpy as np
from numpy import typing as npt
from typing import Any
from numpy import random
from numba import njit


# NEED TO CHANGE np.random.rand()
def midpoint_2d(heightmap, i, j, square_size: float, rand_scale):
    rng = np.random.default_rng(12345)

    # Calculate the side length between points in current square
    size = 2**square_size

    if square_size < 1:
        return heightmap[i : i + size + 1, j : j + size + 1]

    half = size // 2

    # DIAMOND STEP
    # Calculate midpoint of the square
    mid_i = i + half
    mid_j = j + half

    top_left = heightmap[i, j]
    top_right = heightmap[i, j + size]
    bottom_left = heightmap[i + size, j]
    bottom_right = heightmap[i + size, j + size]
    corners = np.array([top_left, top_right, bottom_left, bottom_right])

    heightmap[mid_i, mid_j] = np.mean(corners) + rand_scale * size * (
        rng.random() - 0.5
    )

    heightmap[i, mid_j] = np.mean([top_left, top_right]) + rand_scale * size * (
        rng.random() - 0.5
    )
    # Right side
    heightmap[mid_i, j + size] = np.mean(
        [top_right, bottom_right]
    ) + rand_scale * size * (rng.random() - 0.5)
    # Bottom side
    heightmap[i + size, mid_j] = np.mean(
        [bottom_left, bottom_right]
    ) + rand_scale * size * (rng.random() - 0.5)
    # Left side
    heightmap[mid_i, j] = np.mean([top_left, bottom_left]) + rand_scale * size * (
        rng.random() - 0.5
    )

    # Recursively apply to the four squares
    if square_size > 1:
        midpoint_2d(heightmap, i, j, square_size - 1, rand_scale)  # Top-left
        midpoint_2d(heightmap, i, j + half, square_size - 1, rand_scale)  # Top-right
        midpoint_2d(heightmap, i + half, j, square_size - 1, rand_scale)  # Bottom-left
        midpoint_2d(
            heightmap, i + half, j + half, square_size - 1, rand_scale
        )  # Bottom-right

    return heightmap[i : i + size + 1, j : j + size + 1]


@njit
def random_rays(
    rng: random.Generator,
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    n_rays: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    # rng generator; x0,x1,x2 vertices of triangle; n number of rays

    # Normal vectors
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal)
    normalized = normal / length

    # Orthonormal basis for surface, with third component the normal component
    edge1_normalized = edge1 / np.linalg.norm(edge1)
    edge2_normalized = np.cross(normalized, edge1_normalized)

    matrix = np.array([edge1_normalized, edge2_normalized, normalized]).T

    # Generate n_rays random polar and azimuthal angles.
    theta = np.pi * rng.random(size=n_rays) / 2
    phi = 2 * np.pi * rng.random(size=n_rays)

    ray_angles = np.array(
        [np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)]
    )
    # Transform to 3d coordinates in the standard basis
    ray_vectors = matrix @ ray_angles

    # Uniformly distributed random points inside a triangle.
    barymetric2_array = rng.random(size=n_rays)
    barymetric3_array = rng.random(size=n_rays)
    ray_positions = np.zeros((n_rays, 3))
    for index in range(n_rays):
        random_coord = barycentric_to_cartesian(
            vertex0,
            vertex1,
            vertex2,
            barymetric2_array[index],
            barymetric3_array[index],
        )
        ray_positions[index] = random_coord

    return ray_positions, ray_vectors.T


@njit
def ray_triangle_intersection(
    ray_origin: npt.NDArray[np.floating[Any]],
    ray_direction: npt.NDArray[np.floating[Any]],
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
):
    # Legg til epsilon!?
    edge_a = vertex1 - vertex0
    edge_b = vertex2 - vertex0

    # Calculate determinant
    ray_cross_edge_b = np.cross(ray_direction, edge_b)
    det = np.dot(edge_a, ray_cross_edge_b)

    if det == 0.0:
        return None  # Ray is parallel to the triangle plane

    inv_det = 1.0 / det

    # Calculate distance from vertex0 to ray origin
    origin_to_vertex = ray_origin - vertex0

    u = inv_det * np.dot(origin_to_vertex, ray_cross_edge_b)
    if u < 0.0 or u > 1.0:
        return None

    origin_cross_edge_a = np.cross(origin_to_vertex, edge_a)
    v = inv_det * np.dot(ray_direction, origin_cross_edge_a)

    if v < 0.0 or (u + v) > 1.0:
        return None

    # Compute t to find out where the intersection point is on the line
    t = inv_det * np.dot(edge_b, origin_cross_edge_a)

    if t < 0.0:
        return None  # Intersection is behind the ray start

    # Compute the exact intersection point
    intersection_point = ray_origin + t * ray_direction
    return intersection_point


@njit
def has_ray_triangle_intersected(
    ray_origin: npt.NDArray[np.floating[Any]],
    ray_direction: npt.NDArray[np.floating[Any]],
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
) -> bool:

    intersection = ray_triangle_intersection(
        ray_origin, ray_direction, vertex0, vertex1, vertex2
    )

    return intersection is not None


@njit
def form_factors_raytrace(
    vertices: npt.NDArray[np.floating[Any]],
    triangles_indices: npt.NDArray[np.int64],
    n_rays: int,
    rng: random.Generator,
) -> npt.NDArray[np.floating[Any]]:
    ray_size = np.size(triangles_indices, axis=0)
    form_factor = np.zeros((ray_size, ray_size))
    for i, ray_index_i in enumerate(triangles_indices):
        v0_i = vertices[ray_index_i[0]]
        v1_i = vertices[ray_index_i[1]]
        v2_i = vertices[ray_index_i[2]]
        # generate n rays from random points x in element i, in random directions r
        ray_positions, ray_vectors = random_rays(rng, v0_i, v1_i, v2_i, n_rays)

        # find out which element (if any) each ray hits
        for ray_idx in range(n_rays):
            ray_origin = ray_positions[ray_idx]
            ray_direction = ray_vectors[ray_idx]

            for j, ray_index_j in enumerate(triangles_indices):
                if i == j:  # Skip self-intersection
                    continue

                # Get vertices of element j
                v0_j = vertices[ray_index_j[0]]
                v1_j = vertices[ray_index_j[1]]
                v2_j = vertices[ray_index_j[2]]

                intersection = has_ray_triangle_intersected(
                    ray_origin, ray_direction, v0_j, v1_j, v2_j
                )

                if intersection:
                    # add 1/n to F[i,j] for each ray from i hitting j
                    form_factor[i, j] += 1.0 / n_rays

    return form_factor


def barycentric_to_cartesian(
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    b2: float,
    b3: float,
) -> npt.NDArray[np.floating[Any]]:
    # B2 and B3 are between 0-1
    px = ((1 - b2 - b3) * vertex0[0]) + b2 * vertex1[0] + b3 * vertex2[0]
    py = ((1 - b2 - b3) * vertex0[1]) + b2 * vertex1[1] + b3 * vertex2[1]
    pz = ((1 - b2 - b3) * vertex0[2]) + b2 * vertex1[2] + b3 * vertex2[2]

    return np.array([px, py, pz])


# NOISE THING
# choose your own values:
width = 600
height = 480

rng = np.random.default_rng(12345)

# choose your own values:
grid_x_size = 11
grid_y_size = 9


# generates the grid by
x_grid, y_grid = np.meshgrid(
    np.linspace(0, width, grid_x_size), np.linspace(0, height, grid_y_size)
)


cell_width = width // (grid_x_size - 1)
cell_height = height // (grid_y_size - 1)

# Use rng to generate an (m x n) array of 2d unit vectors, i.e. a (2 x m x n) array
angles = 2 * np.pi * rng.random((grid_x_size, grid_y_size))
vectors = np.array([np.cos(angles), np.sin(angles)])


def easeCurve(t):
    return 3 * t**2 - 2 * t**3


def lerp(a, b, t):
    return a * (1 - t) + b * t


def noise(x, y, x_grid, y_grid, vectors: np.ndarray):
    # Define i and j
    i = np.clip(int(x / cell_width), 0, grid_x_size - 2)
    j = np.clip(int(y / cell_height), 0, grid_y_size - 2)

    # tl = top left, tr = top right, bl = bottom left, br = bottom right
    x_bl = x_grid[j, i]
    y_bl = y_grid[j, i]
    x_br = x_grid[j, i + 1]
    y_br = y_grid[j, i + 1]
    x_tl = x_grid[j + 1, i]
    y_tl = y_grid[j + 1, i]
    x_tr = x_grid[j + 1, i + 1]
    y_tr = y_grid[j + 1, i + 1]

    tx = (x - x_bl) / (x_br - x_bl)
    ty = (y - y_bl) / (y_tl - y_bl)

    # Smooth the noise
    smoothtx = easeCurve(tx)
    smoothty = easeCurve(ty)

    # Find vectors to corners
    v_bl = vectors[:, i, j]
    v_br = vectors[:, i + 1, j]
    v_tl = vectors[:, i, j + 1]
    v_tr = vectors[:, i + 1, j + 1]

    # Compute dot products between corner vectors and distance vectors
    s_bl = np.dot(v_bl, [x - x_bl, y - y_bl])
    s_br = np.dot(v_br, [x - x_br, y - y_br])
    s_tl = np.dot(v_tl, [x - x_tl, y - y_tl])
    s_tr = np.dot(v_tr, [x - x_tr, y - y_tr])

    # Interpolate between dot products
    d_lower = lerp(s_bl, s_br, smoothtx)
    d_upper = lerp(s_tl, s_tr, smoothtx)

    noise = lerp(d_lower, d_upper, smoothty)
    return noise


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


def array_distance_to_plane(
    plane_p: np.ndarray, plane_n: np.ndarray, verticies: np.ndarray
) -> np.ndarray:
    """
    Clipping the triangle against the plane.
    Returns a list of new triangles that are clipped.

    :param plane_n: The normal of the plane.
    :param plane_p: The position of the plane.
    :param vertex_1: The first vertex.
    :param vertex_2: The second vertex.
    :param vertex_3: The third vertex.
    """

    # Make sure the plane is normal
    plane_n = vector_normalize(plane_n)

    # Get distance for each vertex in triangle to plane
    distances = array_distance_point_to_plane(plane_n, plane_p, verticies)

    return distances


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
