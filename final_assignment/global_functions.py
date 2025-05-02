"""
Code for handling vector functions and intersections, raytracing

Author: Mathilde Borsodi Hegna
"""

import numpy as np


# Debug colors
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


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


def triangle_clip_against_plane(
    plane_p: np.ndarray,
    plane_n: np.ndarray,
    vertex_1: np.ndarray,
    vertex_2: np.ndarray,
    vertex_3: np.ndarray,
    distance_1: float,
    distance_2: float,
    distance_3: float,
    color: np.ndarray,
    clip_debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Lists to keep which triangle is outside/inside
    inside: list[np.ndarray] = []
    outside: list[np.ndarray] = []

    # Get distance for each vertex in triangle to plane
    distance_1 = distance_point_to_plane(plane_n, plane_p, vertex_1)
    distance_2 = distance_point_to_plane(plane_n, plane_p, vertex_2)
    distance_3 = distance_point_to_plane(plane_n, plane_p, vertex_3)

    # Adding the different vertices accordingly to if it is inside or outside the plane
    if distance_1 >= 0:
        inside.append(vertex_1)
    else:
        outside.append(vertex_1)
    if distance_2 >= 0:
        inside.append(vertex_2)
    else:
        outside.append(vertex_2)
    if distance_3 >= 0:
        inside.append(vertex_3)
    else:
        outside.append(vertex_3)

    # Clipping depending on which vertices are outside/inside
    if len(inside) == 0:
        # All the vertices are outside
        return (
            np.empty((0, 3)),
            np.empty((0, 3), dtype=int),
            np.empty((0, 3), dtype=int),
        )

    if len(inside) == 3:
        # All the vertices are inside
        # Index will be filled in by the caller
        return (
            np.array([vertex_1, vertex_2, vertex_3]),
            np.array([[0, 1, 2]]),
            np.array([color], dtype=int),
        )

    if len(inside) == 2 and len(outside) == 1:
        # Two vertices inside and one outside

        # Calculate intersect vertex1 clockwise
        intersect_vertex_1 = line_intersect_plane(
            plane_p, plane_n, inside[0], outside[0]
        )
        # Calculate intersect vertex2 clockwise
        intersect_vertex_2 = line_intersect_plane(
            plane_p, plane_n, inside[1], outside[0]
        )

        # Parallel with the plane
        if intersect_vertex_1 is None or intersect_vertex_2 is None:
            return (
                np.empty((0, 3)),
                np.empty((0, 3), dtype=int),
                np.empty((0, 3), dtype=int),
            )

        return (
            np.array([inside[0], inside[1], intersect_vertex_1, intersect_vertex_2]),
            np.array([[0, 1, 2], [1, 3, 2]]),
            np.array(
                [
                    np.array(PINK if clip_debug else color),
                    np.array(WHITE if clip_debug else color),
                ]
            ),
        )

    if len(inside) == 1 and len(outside) == 2:
        # One vertices inside and two outside

        # Calculate intersect vertex1 clockwise
        intersect_vertex_1 = line_intersect_plane(
            plane_p, plane_n, inside[0], outside[0]
        )
        # Calculate intersect vertex2 clockwise
        intersect_vertex_2 = line_intersect_plane(
            plane_p, plane_n, inside[0], outside[1]
        )

        # Parallel with the plane
        if intersect_vertex_1 is None or intersect_vertex_2 is None:
            return (
                np.empty((0, 3)),
                np.empty((0, 3), dtype=int),
                np.empty((0, 3), dtype=int),
            )
        return (
            np.array([inside[0], intersect_vertex_2, intersect_vertex_1]),
            np.array([[0, 1, 2]]),
            np.array([np.array(GREY if clip_debug else color)]),
        )
    # This should never occur
    raise Exception("Something very wrong has happend")


def array_triangle_clip_against_plane(
    plane_p: np.ndarray,
    plane_n: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    clip_debug: bool,
):

    distances = array_distance_to_plane(plane_p, plane_n, vertices)
    list_verticies = []
    list_colors = []
    list_indices = []
    vertex_count = 0

    for triangle_index, triangle in enumerate(triangles):
        index1, index2, index3 = triangle
        vertex1 = vertices[index1]
        vertex2 = vertices[index2]
        vertex3 = vertices[index3]

        distance1: float = distances[index1]  # type: ignore
        distance2: float = distances[index2]  # type: ignore
        distance3: float = distances[index3]  # type: ignore

        clip_verticies, clip_indices, clip_colors = triangle_clip_against_plane(
            plane_p,
            plane_n,
            vertex1,
            vertex2,
            vertex3,
            distance1,
            distance2,
            distance3,
            colors[triangle_index],
            clip_debug,
        )

        list_indices.append(clip_indices + vertex_count)
        vertex_count += len(clip_verticies)
        list_verticies.append(clip_verticies)
        list_colors.append(clip_colors)

    new_indices = np.vstack(list_indices)
    new_verticies = np.vstack(list_verticies)
    new_colors = np.vstack(list_colors)

    return new_indices, new_verticies, new_colors


def array_project(
    vertices: np.ndarray,
    window_width: int,
    window_height: int,
    far: float,
    near: float,
    fov: float,
) -> np.ndarray:
    """
    Project the vertex onto the camera plane and returns it.

    :param vertex: The vertex to project.
    :param window_width: Window width.
    :param window_heigth: Window heigth.
    :return: Transformed vertex.
    """
    # This type of projection I got from watching Youtube videos about 3D projecting and clipping
    aspectratio = window_height / window_width
    # Inverse tan based on FOV
    f_rad = 1 / np.tan(np.radians(fov) / 2)
    q = (far) / (far - near)

    w_array = np.full(len(vertices), 1)[:, np.newaxis]

    homogeneous_vertices = np.hstack([vertices, w_array])

    homogeneous_matrix = np.array(
        [
            [aspectratio * f_rad, 0, 0, 0],
            [0, f_rad, 0, 0],
            [0, 0, q, (-near * q)],
            [0, 0, 1, 0],
        ]
    )

    new_vertices = homogeneous_vertices @ homogeneous_matrix.T

    new_vertices = new_vertices / new_vertices[:, np.newaxis, 3]

    return new_vertices[:, 0:3]


def array_project_screen(
    vertices: np.ndarray,
    window_width: int,
    window_height: int,
    far: float,
    near: float,
    fov: float,
):

    half_width = window_width / 2
    half_height = window_height / 2

    # Projecting all the vertices to the camera plane
    projected = array_project(
        vertices, window_width, window_height, far=far, near=near, fov=fov
    )

    # Centering for viewport adjustments
    center = np.array([1, 1, 0])
    projected += center

    # Scale into view
    scale = np.array([half_width, half_height, 1])
    projected *= scale

    return projected
