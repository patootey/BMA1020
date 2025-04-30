from typing import Any
import numpy as np
from numpy import random
from numpy import typing as npt


def barycentric_to_cartesian(
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    b2: float,
    b3: float,
) -> npt.NDArray[np.floating[Any]]:
    # B1 + B2 + B3 = 1
    # B1 is not needed, can be computed
    px = ((1 - b2 - b3) * vertex0[0]) + b2 * vertex1[0] + b3 * vertex2[0]
    py = ((1 - b2 - b3) * vertex0[1]) + b2 * vertex1[1] + b3 * vertex2[1]
    pz = ((1 - b2 - b3) * vertex0[2]) + b2 * vertex1[2] + b3 * vertex2[2]

    return np.array([px, py, pz])


def normalize_vector(
    vector: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    length = np.linalg.norm(vector)
    return vector / length


def random_rays(
    rng: random.Generator,
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    n_rays: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    normal = normalize_vector(np.cross(edge1, edge2))

    # Normalize vectors
    edge1_normalized = edge1 / np.linalg.norm(edge1)
    # Orthonormal basis for surface, with third component the normal component
    edge2_normalized = np.cross(normal, edge1_normalized)

    matrix = np.array([edge1_normalized, edge2_normalized, normal]).T

    # Generate n_rays random polar and azimuthal angles.
    theta = np.pi * rng.random(size=n_rays) / 2
    phi = 2 * np.pi * rng.random(size=n_rays)
    ray_angles = np.array(
        [np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)]
    )

    # Transform to 3d coordinates in the standard basis
    ray_vectors = matrix @ ray_angles

    # Uniformly distributed random points inside a triangle.
    barycentric1_array = rng.random(size=n_rays)
    barycentric2_array = rng.random(size=n_rays)
    barycentric3_array = rng.random(size=n_rays)

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


def ray_triangle_intersection(
    ray_origin: npt.NDArray[np.floating[Any]],
    ray_direction: npt.NDArray[np.floating[Any]],
    vertex0: npt.NDArray[np.floating[Any]],
    vertex1: npt.NDArray[np.floating[Any]],
    vertex2: npt.NDArray[np.floating[Any]],
    epsilon: float = 1e-10,
) -> npt.NDArray[np.floating[Any]] | None:
    edge_a = vertex1 - vertex0
    edge_b = vertex2 - vertex0

    # Calculate determinant
    ray_cross_edge_b = np.cross(ray_direction, edge_b)
    det = np.dot(edge_a, ray_cross_edge_b)

    if det > -epsilon and det < epsilon:
        return None  # Ray is parallel to the triangle plane

    inv_det = 1.0 / det

    # Calculate distance from vertex0 to ray origin
    origin_to_vertex = ray_origin - vertex0

    u = inv_det * np.dot(origin_to_vertex, ray_cross_edge_b)
    if (u < 0.0 and abs(u) > epsilon) or (u > 1 and abs(u - 1) > epsilon):
        return None

    origin_cross_edge_a = np.cross(origin_to_vertex, edge_a)
    v = inv_det * np.dot(ray_direction, origin_cross_edge_a)

    if (v < 0.0 and abs(v) > epsilon) or ((u + v) > 1.0 and abs(u + v - 1) > epsilon):
        return None

    # Compute t to find out where the intersection point is on the line
    t = inv_det * np.dot(edge_b, origin_cross_edge_a)

    if t <= epsilon:
        return None  # Intersection is behind the ray start

    # Compute the exact intersection point
    intersection_point = ray_origin + t * ray_direction
    return intersection_point


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
