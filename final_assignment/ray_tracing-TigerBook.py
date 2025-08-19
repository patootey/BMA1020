"""
Code that handles formfactor with raytracing, random rays and intersection with rays.

@Author Mathilde Borsodi Hegna
"""

import numpy as np
from numpy import typing as npt
from typing import Any
from numba import njit


@njit
def barycentric_to_cartesian(
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    b2: float,
    b3: float,
) -> npt.NDArray[np.floating[Any]]:
    """
    Converting barycentric coordinates to cartisian.
    B1 is not needed in the formula, can be computed:
        B1 + B2 + B3 = 1

    :param vertex0: First vertex.
    :param vertex1: Second vertex.
    :param vertex2: Third vertex.
    :param b2: Random value.
    :param b3: Random value.
    """
    px = ((1 - b2 - b3) * vertex0[0]) + b2 * vertex1[0] + b3 * vertex2[0]
    py = ((1 - b2 - b3) * vertex0[1]) + b2 * vertex1[1] + b3 * vertex2[1]
    pz = ((1 - b2 - b3) * vertex0[2]) + b2 * vertex1[2] + b3 * vertex2[2]

    return np.array([px, py, pz], dtype=np.float64)


@njit
def normalize_vector(
    vector: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Normalizing the vector.

    :param vector: The vector to normalize.
    """
    length = np.linalg.norm(vector)
    return vector / length


@njit
def random_rays(
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    n_rays: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Generate random rays originating from random points on a triangle surface.

    :param vertex0: First vertex of the triangle.
    :param vertex1: Second vertex of the triangle.
    :param vertex2: Third vertex of the triangle.
    :param n_rays: Number of rays to generate.
    """
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    normal = normalize_vector(np.cross(edge1, edge2))

    # Normalize vectors
    edge1_normalized = edge1 / np.linalg.norm(edge1)
    # Orthonormal basis for surface, with third component the normal component
    edge2_normalized = np.cross(normal, edge1_normalized)

    # matrix = np.zeros((3, 3), dtype=np.float64)
    # matrix = np.array([edge1_normalized, edge2_normalized, normal], dtype=np.float64).T
    # for i in range(3):
    #     matrix[0, i] = edge1_normalized[i]
    #     matrix[1, i] = edge2_normalized[i]
    #     matrix[2, i] = normal[i]
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[:, 0] = edge1_normalized
    matrix[:, 1] = edge2_normalized
    matrix[:, 2] = normal

    # Generate n_rays random polar and azimuthal angles.
    theta = np.pi * np.random.random(n_rays) / 2  # rng.random(size=n_rays) / 2
    phi = (
        2 * np.pi * np.random.random(n_rays)  # rng.random(size=n_rays)
    )  # (np.random.rand(1) * n_rays)  # rng.random(size=n_rays)

    ray_angles = np.zeros((3, n_rays), dtype=np.float64)
    ray_angles[0, :] = np.sin(phi) * np.sin(theta)
    ray_angles[1, :] = np.cos(phi) * np.sin(theta)
    ray_angles[2, :] = np.cos(theta)
    # ray_angles = np.array(
    #     [np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)],
    #     dtype=np.float64,
    # )

    # Transform to 3d coordinates in the standard basis
    ray_vectors = matrix @ ray_angles

    # Uniformly distributed random points inside a triangle.
    barycentric1_array = np.random.random(n_rays)  # rng.random(size=n_rays)
    barycentric2_array = np.random.random(n_rays)  # rng.random(size=n_rays)
    barycentric3_array = np.random.random(n_rays)  # rng.random(size=n_rays)

    # Normalize barycentric coordinates
    sum_bary = barycentric1_array + barycentric2_array + barycentric3_array
    barycentric2_array /= sum_bary
    barycentric3_array /= sum_bary

    ray_positions = np.zeros((n_rays, 3))
    for index in range(n_rays):
        # We only need to use barycentric 2 and 3, one is computed from the others
        random_coord = barycentric_to_cartesian(
            vertex0,
            vertex1,
            vertex2,
            barycentric2_array[index],
            barycentric3_array[index],
        )
        ray_positions[index] = random_coord

    # Transpose ray vectors to get the shape (n_rays,3)
    return ray_positions, ray_vectors.T


@njit
def ray_triangle_intersection(
    ray_origin: npt.NDArray[np.floating[Any]],
    ray_direction: npt.NDArray[np.floating[Any]],
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    epsilon: float = 1e-10,
) -> tuple[npt.NDArray[np.floating[Any]] | None, float]:
    """
    Calculate the intersection point between a ray and a triangle.

    :param ray_origin: Origin point of the ray.
    :param ray_direction: Direction vector of the ray.
    :param vertex0: First vertex of the triangle.
    :param vertex1: Second vertex of the triangle.
    :param vertex2: Third vertex of the triangle.
    :param epsilon: Small value to handle floating point precision issues.
    """
    edge_a = vertex1 - vertex0
    edge_b = vertex2 - vertex0

    # Calculate determinant
    ray_cross_edge_b = np.cross(ray_direction, edge_b)
    det = np.dot(edge_a, ray_cross_edge_b)

    if det > -epsilon and det < epsilon:
        return (None, 0.0)  # Ray is parallel to the triangle plane

    inv_det = 1.0 / det

    # Calculate distance from vertex0 to ray origin
    origin_to_vertex = ray_origin - vertex0

    u = inv_det * np.dot(origin_to_vertex, ray_cross_edge_b)
    if (u < 0.0 and abs(u) > epsilon) or (u > 1 and abs(u - 1) > epsilon):
        return (None, 0.0)

    origin_cross_edge_a = np.cross(origin_to_vertex, edge_a)
    v = inv_det * np.dot(ray_direction, origin_cross_edge_a)

    if (v < 0.0 and abs(v) > epsilon) or ((u + v) > 1.0 and abs(u + v - 1) > epsilon):
        return (None, 0.0)

    # Compute t to find out where the intersection point is on the line
    t = inv_det * np.dot(edge_b, origin_cross_edge_a)

    if t <= epsilon:
        return (None, 0.0)  # Intersection is behind the ray start

    # Compute the exact intersection point
    intersection_point = ray_origin + t * ray_direction
    return intersection_point, t  # intersection_point


@njit
def has_ray_triangle_intersected(
    ray_origin: npt.NDArray[np.floating[Any]],
    ray_direction: npt.NDArray[np.floating[Any]],
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
) -> tuple[bool, float]:
    """
    Check if a ray intersects with a triangle and return the distance to the intersection.

    :param ray_origin: Origin point of the ray.
    :param ray_direction: Direction vector of the ray.
    :param vertex0: First vertex of the triangle.
    :param vertex1: Second vertex of the triangle.
    :param vertex2: Third vertex of the triangle.
    """
    intersection, t = ray_triangle_intersection(
        ray_origin, ray_direction, vertex0, vertex1, vertex2
    )
    return intersection is not None, t


@njit
def form_factors_raytrace(
    vertices: npt.NDArray[np.floating[Any]],
    triangles_indices: npt.NDArray[np.int64],
    normals: np.ndarray,
    # centroids: np.ndarray,
    n_rays: int,
) -> npt.NDArray[np.floating[Any]]:
    """
    Calculate form factors between triangles.

    :param verticies: Array of verticies.
    :param triangles_indices: Array of triangle indices.
    :param normals: Array of normals.
    :param n_rays: Number of rays to calculate.
    """
    ray_size = triangles_indices.shape[0]
    form_factor = np.zeros((ray_size, ray_size), dtype=np.float64)
    for i, ray_index_i in enumerate(triangles_indices):
        v0_i = vertices[ray_index_i[0]]
        v1_i = vertices[ray_index_i[1]]
        v2_i = vertices[ray_index_i[2]]
        # generate n rays from random points x in element i, in random directions r
        ray_positions, ray_vectors = random_rays(v0_i, v1_i, v2_i, n_rays)

        # find out which element (if any) each ray hits
        for ray_idx in range(n_rays):
            ray_origin = ray_positions[ray_idx]
            ray_direction = ray_vectors[ray_idx]

            min_occlusion_distance = np.inf
            closest_element = -1
            for j, ray_index_j in enumerate(triangles_indices):
                if i == j:  # Skip self-intersection
                    continue
                # Get vertices of element j
                v0_j = vertices[ray_index_j[0]]
                v1_j = vertices[ray_index_j[1]]
                v2_j = vertices[ray_index_j[2]]

                dot_product = np.dot(ray_direction, normals[j])
                if dot_product >= 0:
                    continue

                hit, distance = has_ray_triangle_intersected(
                    ray_origin, ray_direction, v0_j, v1_j, v2_j
                )

                if hit and distance < min_occlusion_distance:
                    min_occlusion_distance = distance
                    closest_element = j

            if closest_element != -1:
                form_factor[i, closest_element] += 1.0 / n_rays

    return form_factor


@njit
def get_visible_triangles(
    camera_position: np.ndarray,
    triangle_indices: np.ndarray,
    vertices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """
    Check if a triangle is visible from the camera position and direction.

    :param camera_position: Position of the camera.
    :param camera_direction: Direction of the camera.
    :param triangle_indices: Array of all triangles.
    :param vertices: Array of all vertices.
    :param centroids: Array of centroids of all triangles.
    :param normals: Array of normals of all triangles.
    """
    # Pre fill
    num_triangles = len(triangle_indices)
    visible = np.ones(num_triangles, dtype=np.bool_)
    directions = np.zeros((num_triangles, 3), dtype=np.float64)
    distances = np.zeros(num_triangles, dtype=np.float64)

    # Pre-compute centroids and directions for all triangles
    directions = centroids - camera_position
    for i in range(num_triangles):
        # Calculate direction and distance
        # Had to use for loop because of numba limitation with norm and axis
        distances[i] = np.linalg.norm(directions[i])
        directions[i] = directions[i] / distances[i]  # Normalize

    # Sort triangles by distance, the same as for distance check for which triangle to show first
    sorted_indices = np.argsort(distances)

    # Check visibility for each triangle
    for i in range(num_triangles):
        triangle_index = sorted_indices[i]

        # If already marked as invisible, skip
        if not visible[triangle_index]:
            continue

        # Check the normals to be drawn, if points other way, skip
        if np.dot(normals[triangle_index], directions[triangle_index]) >= 0:
            visible[triangle_index] = False
            continue

        # Get triangle vertices for accurate intersection test
        indices = triangle_indices[triangle_index]
        vec0 = vertices[indices[0]]
        vec1 = vertices[indices[1]]
        vec2 = vertices[indices[2]]

        # Get data for current triangle
        direction = directions[triangle_index]

        # Check if this triangle occludes any farther triangles
        for j in range(i + 1, num_triangles):
            occluded_index = sorted_indices[j]

            # If already marked as invisible, skip
            if not visible[occluded_index]:
                continue

            # Vector from camera to farther triangle
            far_direction = directions[occluded_index]

            # Angle between the two vectors
            angle = np.dot(direction, far_direction)

            # This part I got help from AI (angle check)
            # If the angle is small enough, the triangles might occlude each other
            if angle > 0.99:  # Threshold value
                # Check if the ray to the farther triangle intersects this triangle
                hit, hit_dist = has_ray_triangle_intersected(
                    camera_position, far_direction, vec0, vec1, vec2
                )

                # If there's a hit and it's closer than the farther triangle
                if hit and hit_dist < distances[occluded_index]:
                    visible[occluded_index] = False

    return visible
