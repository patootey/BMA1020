"""
This code generates a cube, rotates it and displays it on the screen.

The controls:
    To go upwards press: SPACE
    To go downwards press: CTRL
    Normal movement: Forward(W), Left(A), Back(S) & Right(D)
    Turn the cube in the X-axis: Left & Right arrow keys
    Turn the cube in the Y-axis: Up & Down arrow keys
    Turn the cube in the Z-axis: Q & E keys
Debug:
    To activate wireframe: F1
    To activate debug clipping: F2
    To change FOV: F3 & F4

@Author Mathilde Borsodi Hegna
"""

from contextlib import contextmanager
from datetime import datetime
from global_functions import (
    distance_point_to_plane,
    line_intersect_plane,
    vector_normalize,
    array_distance_to_plane,
)
import numpy as np
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternions import Quaternion
from rgb_converting import LMS_to_RGB

# Global variables
g_window_width = 800
g_window_height = 600
g_window = Window(g_window_width, g_window_height)

g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)

FRAME_TIME: float = 1 / 60

# Debug colors
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


class Ray_tracing_model_3d:
    vertices: np.ndarray
    "Array of vertex arrays"
    temporary_vertices: np.ndarray
    "Temporary copy of vertices for transformation"
    triangle_indices: np.ndarray
    "Array of triangle arrays"
    colors: np.ndarray
    "Array of color arrays"
    fov: float = 100
    "Field of view"
    far: float = 1000
    "Far plane for camera"
    near: float = 0.1
    "Near plane for camera"
    batch: Batch
    "Batch of the pyglet shapes to draw in one call"
    triangles: list[Triangle]
    "Store all pyglet triangles that are created"
    clip_debug: bool = False
    "Debugging tool to show when the triangles are clipped"
    wireframe: bool = False
    "Debugging tool to show all the triangles in the cube"
    lines: list[Line]
    "Store all pyglet lines that are created to show a wireframe"
    rotation_angle_x: float = 0
    "The rotation angle of the cube around the X-axis"
    rotation_angle_y: float = 0
    "The rotation angle of the cube around the Y-axis"
    rotation_angle_z: float = 0
    "The rotation angle of the cube around the Z-axis"
    position: np.ndarray
    "The position of the cube"
    emit_array: np.ndarray
    "Array of emission values for triangles"
    reflect_array: np.ndarray
    "Array of reflection values for triangles"
    normal_array: np.ndarray
    "Array of all triangle normals"
    area_array: np.ndarray
    "Array of all areas"
    form_factor_array: np.ndarray
    "Array of all form factors"

    def __init__(
        self,
        vertices: np.ndarray,
        triangle_indices: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:
        # Previous named: nodes
        self.vertices = vertices
        self.temporary_vertices = np.array([])
        # Previous named: elements
        self.triangle_indices = triangle_indices
        self.colors = np.zeros((len(triangle_indices), 3), dtype=np.int32)
        self.batch = Batch()
        # Changed from a dictionary to a list
        self.triangles = []
        # NEWW
        self.next_triangle_index = 0
        # NEWWWW
        self.next_line_index = 0
        self.lines = []
        self.position = np.array([0, 0, 0])
        self.emit_array = emit
        self.reflect_array = reflect
        self.normal_array: np.ndarray = np.array([])
        self.area_array: np.ndarray = np.array([])
        self.form_factor_array: np.ndarray = np.array([])

    def prepare(self) -> None:
        """
        Preparing for a new loop.
        Initializing temporary properties.
        """
        # Saving the old coordianates of vertices
        self.temporary_vertices = self.vertices.copy()

        # Dont show the triangles
        for triangle in self.triangles:
            triangle.visible = False
        # Keep track of the next triangle
        self.next_triangle_index = 0

        # Dont show the lines
        for line in self.lines:
            line.visible = False
        # Keep track of the next line
        self.next_line_index = 0

    def add_vertex(self, vertex: np.ndarray) -> None:  # 5 points
        """
        Add a new vertex.

        :param vertex: Vertex with x,y,z coordinate.
        """
        # Previously add_node
        # Add a vertex to the model
        self.vertices = np.vstack((self.vertices, np.array([vertex])))

    def delete_vertex(self, index: int) -> None:  # 5 points
        """
        Delete vertex at given index, will also delete all triangles that uses the vertex.

        :param index: Index of the vertex.
        """
        # Previously named: delete_node
        # Count for how many triangles that are deleted
        deleted_count = 0
        # Need to delete the triangles pointing to the vertex
        for index_triangle, triangle in enumerate(self.triangle_indices):
            if triangle[0] == index or triangle[1] == index or triangle[2] == index:
                self.delete_triangle(index_triangle - deleted_count)
                deleted_count += 1
            else:
                # Adjusting the vertex index in each triangle
                if triangle[0] > index:
                    triangle[0] -= 1
                if triangle[1] > index:
                    triangle[1] -= 1
                if triangle[2] > index:
                    triangle[2] -= 1

                # Copy to the right index
                self.triangle_indices[index_triangle - deleted_count] = triangle

        # Remove a vertex from the model
        self.vertices = np.delete(self.vertices, index, axis=0)

    def add_triangle(
        self,
        triangle: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:  # 5 points
        """
        Add triangle with a emit and reflect value.

        :param triangle: Index of vertices of the triangle.
        :param emit: Emit value of the triangle.
        :param reflect: Reflect value of the triangle.
        """
        # Previously named: add_element
        # And removed back parameter
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))
        self.emit_array = np.vstack((self.emit_array, np.array([emit])))
        self.reflect_array = np.vstack((self.reflect_array, np.array([reflect])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        """
        Delete a triangle at given index.

        :param index: Index of the triangle.
        """
        # Previously named: delete_element
        self.triangle_indices = np.delete(self.triangle_indices, index, axis=0)
        self.emit_array = np.delete(self.emit_array, index, axis=0)
        self.reflect_array = np.delete(self.reflect_array, index, axis=0)

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotate all the vertices by a quaternion.

        :param quaternion: Quaternion object
        """
        # up to 25 points, including the quaternion code
        self.temporary_vertices = quaternion.rotate_vectors(self.temporary_vertices)

    def translate(self, vector: np.ndarray) -> None:  # 10 points
        """
        Translating all vertices by the vector.

        :param vector: Vector direction.
        """
        if vector.ndim != 1:
            raise ValueError("Vector must be 1D")
        if len(vector) != 3:
            raise ValueError("Vector length must be 3")

        for i in range(len(self.temporary_vertices)):
            self.temporary_vertices[i] = self.temporary_vertices[i] + vector

    def array_project(
        self, vertices: np.ndarray, window_width: int, window_height: int
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
        f_rad = 1 / np.tan(np.radians(self.fov) / 2)
        q = (self.far) / (self.far - self.near)

        w_array = np.full(len(vertices), 1)[:, np.newaxis]

        homogeneous_vertices = np.hstack([vertices, w_array])

        homogeneous_matrix = np.array(
            [
                [aspectratio * f_rad, 0, 0, 0],
                [0, f_rad, 0, 0],
                [0, 0, q, (-self.near * q)],
                [0, 0, 1, 0],
            ]
        )

        new_vertices = homogeneous_vertices @ homogeneous_matrix.T

        new_vertices = new_vertices / new_vertices[:, np.newaxis, 3]

        return new_vertices[:, 0:3]

    def array_triangle_clip_against_plane(
        self,
        plane_p: np.ndarray,
        plane_n: np.ndarray,
        vertices: np.ndarray,
        triangles: np.ndarray,
        colors: np.ndarray,
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

            clip_verticies, clip_indices, clip_colors = (
                self.triangle_clip_against_plane(
                    plane_p,
                    plane_n,
                    vertex1,
                    vertex2,
                    vertex3,
                    distance1,
                    distance2,
                    distance3,
                    colors[triangle_index],
                )
            )

            list_indices.append(clip_indices + vertex_count)
            vertex_count += len(clip_verticies)
            list_verticies.append(clip_verticies)
            list_colors.append(clip_colors)

        new_indices = np.vstack(list_indices)
        new_verticies = np.vstack(list_verticies)
        new_colors = np.vstack(list_colors)

        return new_indices, new_verticies, new_colors

    def triangle_clip_against_plane(
        self,
        plane_p: np.ndarray,
        plane_n: np.ndarray,
        vertex_1: np.ndarray,
        vertex_2: np.ndarray,
        vertex_3: np.ndarray,
        distance_1: float,
        distance_2: float,
        distance_3: float,
        color: np.ndarray,
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
                np.array(
                    [inside[0], inside[1], intersect_vertex_1, intersect_vertex_2]
                ),
                np.array([[0, 1, 2], [1, 3, 2]]),
                np.array(
                    [
                        np.array(PINK if self.clip_debug else color),
                        np.array(WHITE if self.clip_debug else color),
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
                np.array([np.array(GREY if self.clip_debug else color)]),
            )
        # This should never occur
        raise Exception("Something very wrong has happend")

    def array_project_screen(
        self,
        vertices: np.ndarray,
        window_width: int,
        window_height: int,
    ):

        half_width = window_width / 2
        half_height = window_height / 2

        # Projecting all the vertices to the camera plane
        projected = self.array_project(vertices, window_width, window_height)

        # Centering for viewport adjustments
        center = np.array([1, 1, 0])
        projected += center

        # Scale into view
        scale = np.array([half_width, half_height, 1])
        projected *= scale

        return projected

    def build_triangle(
        self,
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        vertex3: np.ndarray,
        color: np.ndarray,
    ) -> None:
        """
        Building one triangle shape by projecting the vertex to the camera and transform to the viewport.
        Then adds the triangle to the active triangle list.
        If wireframe is on, it will draw lines on top of the triangle.

        :param trigon: The trigon object.
        :param window_width: Window width.
        :param window_heigth: Window heigth.
        """
        if self.next_triangle_index < len(self.triangles):
            triangle = self.triangles[self.next_triangle_index]
            triangle.x = vertex1[0]
            triangle.y = vertex1[1]
            triangle.x2 = vertex2[0]
            triangle.y2 = vertex2[1]
            triangle.x3 = vertex3[0]
            triangle.y3 = vertex3[1]
            triangle.color = tuple(color)
            triangle.visible = True
            self.next_triangle_index += 1
        else:
            # If not, draw new ones
            triangle = Triangle(
                vertex1[0],
                vertex1[1],
                vertex2[0],
                vertex2[1],
                vertex3[0],
                vertex3[1],
                color=tuple(color),
                batch=self.batch,
            )
            self.triangles.append(triangle)
            self.next_triangle_index += 1

        # Debugging helpers
        if self.wireframe:
            if self.next_line_index < len(self.lines):
                # GJØR DET BARE FOR EN LINE, MÅ FOR ALLE TRE
                line = self.lines[self.next_line_index]
                line.x = vertex1[0]
                line.y = vertex1[1]
                line.x2 = vertex2[0]
                line.y2 = vertex2[1]
                line.visible = True
                self.next_line_index += 1
            else:
                self.lines.append(
                    Line(
                        vertex1[0], vertex1[1], vertex2[0], vertex2[1], batch=self.batch
                    )
                )
                self.lines.append(
                    Line(
                        vertex2[0], vertex2[1], vertex3[0], vertex3[1], batch=self.batch
                    )
                )
                self.lines.append(
                    Line(
                        vertex1[0], vertex1[1], vertex3[0], vertex3[1], batch=self.batch
                    )
                )
                self.next_line_index += 1

    def build_triangles(self, window_width: int, window_height: int) -> None:
        """
        Builds all the triangles for the viewport.
        The triangles will be rendered by pyglet.

        :param window_width: Window width.
        :param window_height: Window heigth.
        """
        # Originally named: draw
        # Draw the model to the screen
        new_indices, new_vertices, new_colors = self.array_triangle_clip_against_plane(
            plane_p=np.array([0, 0, 0.1]),
            plane_n=np.array([0, 0, 1]),
            vertices=self.temporary_vertices,
            triangles=self.triangle_indices,
            colors=self.colors,
        )

        projected_vertices = self.array_project_screen(
            vertices=new_vertices,
            window_width=window_width,
            window_height=window_height,
        )

        # Rebuild the triangles for each edge
        for triangle_index in range(len(new_indices)):
            color = new_colors[triangle_index]
            triangle_array: np.ndarray = new_indices[triangle_index]
            vertex1_index, vertex2_index, vertex3_index = triangle_array
            # Make sure the vertices does not get the same coords/index
            vertex1 = projected_vertices[vertex1_index]
            vertex2 = projected_vertices[vertex2_index]
            vertex3 = projected_vertices[vertex3_index]

            self.build_triangle(
                vertex1=vertex1,
                vertex2=vertex2,
                vertex3=vertex3,
                color=color,
            )

    @classmethod  # 10 points
    def model_rect(
        cls,
        x_min: float,
        x_max: float,
        x_divisions: int,
        y_min: float,
        y_max: float,
        y_divisions: int,
        emission: np.ndarray = np.array([0, 0, 0]),
        reflect: np.ndarray = np.array([0.5, 0.5, 0.5]),
        translation_vector: np.ndarray = np.array([0, 0, 0]),
        rotation_axis: np.ndarray = np.array([0, 0, 1]),
        rotation_angle: float = 0,
        flip_normals: bool = False,
        # NEW
        midpoint=False,
        midpoint_scale=1.0,
    ) -> "Ray_tracing_model_3d":
        """
        Create a rectangular grid model in the plane and apply tranformations.

        :param x_min:  Minimum X coordinate.
        :param x_max:  Maximum X coordinate.
        :param x_divisions: Number of divisions along X-axis.
        :param y_min:  Minimum y coordinate.
        :param y_max:  Maximum y coordinate.
        :param y_divisions: Number of divisions along Y-axis.
        :param emission: Emission for each triangle.
        :param reflect: Reflectance for each triangle.
        :param translation_vector: 3D translation vector.
        :param rotation_axis: Axis around which to rotate the model.
        :param rotation_angle: Angle to rotate (in radians).
        :param flip_normals: If True, flip the normals direction.
        :return: A new instance of Light_model_3d representing the model.
        """

        x_coords = np.linspace(x_min, x_max, x_divisions)
        y_coords = np.linspace(y_min, y_max, y_divisions)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        flat_xy = np.transpose(
            np.vstack(
                (
                    grid_x.reshape(y_divisions * x_divisions),
                    grid_y.reshape(y_divisions * x_divisions),
                )
            )
        )
        z_values = np.zeros((x_divisions * y_divisions, 1))
        vertices = np.hstack((flat_xy, z_values))

        # Create triangles by connecting grid points
        triangle_indices = []

        for row in range(y_divisions - 1):
            for col in range(x_divisions - 1):
                # The grid is laid out row by row
                top_left = row * x_divisions + col
                top_right = top_left + 1
                bottom_left = (row + 1) * x_divisions + col
                bottom_right = bottom_left + 1

                if not flip_normals:
                    # Clockwise
                    triangle_indices.append([top_left, top_right, bottom_left])
                    triangle_indices.append([bottom_left, top_right, bottom_right])
                else:
                    # Flipped counter-clockwise
                    triangle_indices.append([bottom_left, top_right, top_left])
                    triangle_indices.append([bottom_right, top_right, bottom_left])

        triangle_indices = np.array(triangle_indices)

        quaternion_rotation = Quaternion.from_rotation(
            np.array(rotation_axis), rotation_angle
        )

        # Apply rotation
        rotated_vertices = quaternion_rotation.rotate_vectors(vertices)

        # Apply translation
        translated_vertices = rotated_vertices + np.array(translation_vector)

        # Create arrays for emission and reflection values for each triangle
        emit_array = np.array([emission for _ in range(len(triangle_indices))])
        reflect_array = np.array([reflect for _ in range(len(triangle_indices))])

        return Ray_tracing_model_3d(
            translated_vertices, triangle_indices, emit_array, reflect_array
        )

    def midpoint(self, A, i, j, rand_scale):
        # your method can call your routine from assignment 9.
        pass

    def update_geometry(self) -> None:  # 5 points
        """
        Compute the normals and areas of each triangle in the plane.
        Used some from assignment 8
        """
        num_triangles = len(self.triangle_indices)

        self.normal_array = np.zeros((num_triangles, 3))
        self.area_array = np.zeros(num_triangles)

        for i in range(num_triangles):
            # Get indices of the three vertices
            index_vertex = self.triangle_indices[i]

            # Get the vertices
            vertex_1 = self.vertices[index_vertex[0]]
            vertex_2 = self.vertices[index_vertex[1]]
            vertex_3 = self.vertices[index_vertex[2]]

            vector_1 = vertex_2 - vertex_1
            vector_2 = vertex_3 - vertex_1

            # Calculate normal using cross product
            normal = np.cross(vector_1, vector_2)

            # Normalize the normal vector
            normalize = np.linalg.norm(normal)
            if normalize > 0:
                normal = normal / normalize

            area = normalize / 2

            # Store results
            self.normal_array[i] = normal
            self.area_array[i] = area

    def update_form_factors(self) -> None:  # 5 points
        """
        Compute the form factor matrix for all triangle pairs.
        Used some from assignment 8

        Uses the formula:
            Fij = (1/Ai) * ∫∫ (cos(θi)cos(θj)/(π*r²)) dAidAj
        """
        num_triangles = len(self.triangle_indices)
        self.form_factor_array = np.zeros((num_triangles, num_triangles))

        centroids = np.zeros((num_triangles, 3))
        for i in range(num_triangles):
            centroids[i] = np.mean(self.vertices[self.triangle_indices[i]], axis=0)

        centroids_i = centroids[:, np.newaxis, :]
        centroids_j = centroids[np.newaxis, :, :]

        r_vectors = centroids_j - centroids_i

        r_norms = np.linalg.norm(r_vectors, axis=2)

        # Create a mask for diagonal elements to avoid division by zero
        # HJELP AV AI
        non_diagonal = np.ones((num_triangles, num_triangles), dtype=bool)
        np.fill_diagonal(non_diagonal, False)

        # Only process non-diagonal elements
        i_indices, j_indices = np.where(non_diagonal)

        valid_r_vectors = r_vectors[i_indices, j_indices]
        valid_r_norms = r_norms[i_indices, j_indices]

        # Compute unit direction vectors
        valid_r_units = valid_r_vectors / valid_r_norms[:, np.newaxis]

        normals_i = self.normal_array[i_indices]
        dot_ni = np.sum(valid_r_units * normals_i, axis=1)

        normals_j = self.normal_array[j_indices]
        dot_nj = np.sum(-valid_r_units * normals_j, axis=1)

        valid_dots = (dot_ni > 0) & (dot_nj > 0)

        valid_i = i_indices[valid_dots]
        valid_j = j_indices[valid_dots]

        final_dot_ni = dot_ni[valid_dots]
        final_dot_nj = dot_nj[valid_dots]
        final_r_norms = valid_r_norms[valid_dots]

        computed_factors = (
            self.area_array[valid_j]
            * (final_dot_ni * final_dot_nj)
            / (np.pi * final_r_norms**2)
        )

        self.form_factor_array[valid_i, valid_j] = computed_factors

    def find_luminances(self, num_iterations: int) -> np.ndarray:
        """
        Compute the final luminance (radiosity) values for each triangle
        using an iterative radiosity method.
        Used some from assignment 8.

        :param num_iterations: Number of iterations for the radiosity computation.
        :return: A array of RGB luminance values.
        """

        # num_triangles = len(self.triangle_indices)
        radiosity_array = self.emit_array.copy()
        initial_emission_array = self.emit_array.copy()

        # Create a mask to exclude diagonal elements (i==j)
        # HJELP AV AI
        mask_non_diagonal = np.ones(
            (len(self.triangle_indices), len(self.triangle_indices)), dtype=bool
        )
        np.fill_diagonal(mask_non_diagonal, False)

        for _ in range(num_iterations):
            prev_radiosity = radiosity_array.copy()

            radiosity_array = initial_emission_array.copy()

            prev_radiosity_expanded = prev_radiosity[np.newaxis, :, :]
            form_factors_expanded = self.form_factor_array[:, :, np.newaxis]
            reflect_expanded = self.reflect_array[:, np.newaxis, :]
            mask_non_diagonal_expanded = mask_non_diagonal[:, :, np.newaxis]

            light_contributions = form_factors_expanded * prev_radiosity_expanded
            reflected_light = reflect_expanded * light_contributions

            # only the ones that are not diagonal i == j
            reflected_light = reflected_light * mask_non_diagonal_expanded

            total_reflected = np.sum(reflected_light, axis=1)
            radiosity_array += total_reflected

        # Clip values to a valid range and return
        luminances = np.clip(radiosity_array, 0, 1)
        return luminances

    def find_colors(self, num_iterations: int) -> np.ndarray:  # 10 points
        """
        Convert computed luminance values into gamma-corrected RGB color values.
        Used some from assignment 8

        :param num_iterations: Number of iterations for the radiosity calculation
        :return: Array of RGB colors for each triangle
        """
        # Calculate LMS luminances
        lms_luminances = self.find_luminances(num_iterations)

        # Initialize RGB colors array
        num_triangles = len(self.triangle_indices)
        rgb_colors = np.zeros((num_triangles, 3), dtype=np.int32)

        # Convert LMS to RGB for each triangle
        for index in range(num_triangles):
            rgb_colors[index] = LMS_to_RGB(lms_luminances[index])

        return rgb_colors

    def add_model(self, other: "Ray_tracing_model_3d") -> None:  # 10 points
        """
        Merge another Light_model_3d instance into the current model.

        :param other: Another Light_model_3d instance to merge with this one.
        """
        # Store the current number of vertices to offset the indices from the other model
        vertex_offset = len(self.vertices)

        # Append vertices
        self.vertices = np.vstack((self.vertices, other.vertices))

        # Offset triangle indices and append
        offset_indices = other.triangle_indices + vertex_offset
        self.triangle_indices = np.vstack((self.triangle_indices, offset_indices))

        # Append emit and reflect values
        self.emit_array = np.vstack((self.emit_array, other.emit_array))
        self.reflect_array = np.vstack((self.reflect_array, other.reflect_array))


# Used for printing out how much time a function or code takes
@contextmanager
def measure_duration(text: str, is_active: bool):
    if is_active:
        start_time = datetime.now()
        yield
        duration = datetime.now() - start_time
        print(f"{text} {duration}")
    else:
        yield


def main() -> None:
    cube = Ray_tracing_model_3d(
        vertices=np.empty((0, 3)),
        triangle_indices=np.empty((0, 3), dtype=int),
        emit=np.empty((0, 3)),
        reflect=np.empty((0, 3)),
    )

    cube_size = 125
    cube_divisions = 10
    offset = 75
    # Front wall
    front_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.5, 0.1, 0.7]),
        translation_vector=np.array([0, 0, cube_size / 2 + offset]),  # Front position
        rotation_axis=np.array([0, 0, 1]),
        rotation_angle=0,
        flip_normals=True,
    )
    cube.add_model(front_wall)

    # Back wall
    back_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.0, 0.5, 0.5]),
        translation_vector=np.array([0, 0, -cube_size / 2]),  # Back position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi,  # 180 degree rotation
        flip_normals=True,
    )
    cube.add_model(back_wall)

    # Right wall
    right_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2 + offset,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.8, 0.5, 0.3]),
        translation_vector=np.array([cube_size / 2, 0, 0]),  # Right position
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=-np.pi / 2,  # -90 degree rotation
    )
    cube.add_model(right_wall)

    # Left wall
    left_wall = cube.model_rect(
        -cube_size / 2 - offset,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.1, 0.3, 0.5]),
        translation_vector=np.array([-cube_size / 2, 0, 0]),  # Left position
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.pi / 2,  # 90 degree rotation
    )
    cube.add_model(left_wall)

    # Bottom wall
    bottom_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2 + offset,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.1, 0.6, 0.1]),
        translation_vector=np.array([0, -cube_size / 2, 0]),  # Bottom position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,  # 90 degree rotation
        flip_normals=True,
    )
    cube.add_model(bottom_wall)

    # Top wall
    top_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2 - offset,
        cube_size / 2,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.5, 0.1, 0.2]),
        translation_vector=np.array([0, cube_size / 2, 0]),  # Top position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=-np.pi / 2,  # -90 degree rotation
        flip_normals=True,
    )
    cube.add_model(top_wall)

    # USED AI TO GENRATE COLUMNS
    column_size = 25
    column_height = cube_size - 5
    column_division = 2
    reflectance_gray = np.array([0.2, 0.2, 0.2])  # Medium gray

    # Offsets to place columns toward corners
    column_offset = cube_size / 2 - column_size

    # 4 corner positions
    column_positions = [
        np.array([column_offset, 0, column_offset]),  # Front right
        np.array([-column_offset, 0, column_offset]),  # Front left
        np.array([column_offset, 0, -column_offset]),  # Back right
        np.array([-column_offset, 0, -column_offset]),  # Back left
    ]

    for pos in column_positions:
        for i in range(4):
            angle = i * np.pi / 2
            face_translation = pos + np.array(
                [column_size / 2 * np.sin(angle), 0, column_size / 2 * np.cos(angle)]
            )

            column_face = cube.model_rect(
                -column_size / 2,
                column_size / 2,
                column_division,
                -column_height / 2,
                column_height / 2,
                column_division,
                emission=np.array([0.0, 0.0, 0.0]),
                reflect=reflectance_gray,
                translation_vector=face_translation,
                rotation_axis=np.array([0, 1, 0]),
                rotation_angle=-angle,
            )
            cube.add_model(column_face)

        # Now add top and bottom faces
        bottom_face = cube.model_rect(
            -column_size / 2,
            column_size / 2,
            column_division,
            -column_size / 2,
            column_size / 2,
            column_division,
            emission=np.array([0.0, 0.0, 0.0]),
            reflect=reflectance_gray,
            translation_vector=pos + np.array([0, -column_height / 2, 0]),
            rotation_axis=np.array([1, 0, 0]),
            rotation_angle=np.pi / 2,
            flip_normals=True,
        )
        cube.add_model(bottom_face)

        top_face = cube.model_rect(
            -column_size / 2,
            column_size / 2,
            column_division,
            -column_size / 2,
            column_size / 2,
            column_division,
            emission=np.array([0.0, 0.0, 0.0]),
            reflect=reflectance_gray,
            translation_vector=pos + np.array([0, column_height / 2, 0]),
            rotation_axis=np.array([1, 0, 0]),
            rotation_angle=-np.pi / 2,
            flip_normals=False,
        )
        cube.add_model(top_face)

    light_size = 30
    light_division = 2

    light_1 = cube.model_rect(
        -cube_size / 5 - light_size / 2,
        -cube_size / 5 + light_size / 2,
        light_division,
        -cube_size / 5 - light_size / 2,
        -cube_size / 5 + light_size / 2,
        light_division,
        emission=np.array([10.0, 10.0, 10.0]),
        reflect=np.array([0.0, 0.0, 0.0]),
        translation_vector=np.array([0, cube_size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
    )
    cube.add_model(light_1)

    light_2 = cube.model_rect(
        cube_size / 5 - light_size / 2,
        cube_size / 5 + light_size / 2,
        light_division,
        cube_size / 5 - light_size / 2,
        cube_size / 5 + light_size / 2,
        light_division,
        emission=np.array([15.0, 15.0, 15.0]),
        reflect=np.array([0.0, 0.0, 0.0]),
        translation_vector=np.array([0, cube_size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
    )
    cube.add_model(light_2)
    # light_size = 30
    # light_division = 4

    # # Center ceiling light
    # center_light = cube.model_rect(
    #     -light_size / 2,
    #     light_size / 2,
    #     light_division,
    #     -light_size / 2,
    #     light_size / 2,
    #     light_division,
    #     emission=np.array([10.0, 10.0, 10.0]),  # Bright white light
    #     reflect=np.array([0.0, 0.0, 0.0]),
    #     translation_vector=np.array(
    #         [0, cube_size / 2 - 5, 0]
    #     ),  # Slightly below ceiling
    #     rotation_axis=np.array([1, 0, 0]),
    #     rotation_angle=np.pi / 2,  # Pointing down
    # )
    # cube.add_model(center_light)

    print("Update geometry...")
    cube.update_geometry()

    with measure_duration("Form factor duration: ", True):
        print("Calculating form factors...")
        cube.update_form_factors()

    with measure_duration("Color duration: ", True):
        print("Calculating light luminances...")
        cube.colors = cube.find_colors(num_iterations=25)

    show_duration = False
    show_triangle_count = False
    if show_triangle_count:
        triangle_count_label = pyglet.text.Label(
            "",
            font_name="Times New Roman",
            font_size=36,
            x=g_window_width // 2,
            y=g_window_height // 2,
            anchor_x="center",
            anchor_y="center",
        )

    # Initial preparation in case the drawing function is called
    cube.prepare()

    # Draw to the screen
    @g_window.event
    def on_draw() -> None:
        g_window.clear()
        cube.batch.draw()

        if show_triangle_count:
            triangle_count_label.draw()

    def update(dt: float) -> None:
        speed = 100 * dt
        rotation_speed = 1.0 * dt

        if g_keys[key.LSHIFT]:
            speed *= 2
        if g_keys[key.LALT]:
            rotation_speed *= 3

        if g_keys[key.LEFT]:
            cube.rotation_angle_y += rotation_speed
        if g_keys[key.RIGHT]:
            cube.rotation_angle_y -= rotation_speed
        if g_keys[key.UP]:
            cube.rotation_angle_x += rotation_speed
        if g_keys[key.DOWN]:
            cube.rotation_angle_x -= rotation_speed
        if g_keys[key.Q]:
            cube.rotation_angle_z += rotation_speed
        if g_keys[key.E]:
            cube.rotation_angle_z -= rotation_speed

        qx = Quaternion.from_rotation(np.array([1, 0, 0]), cube.rotation_angle_x)
        qy = Quaternion.from_rotation(np.array([0, 1, 0]), cube.rotation_angle_y)
        qz = Quaternion.from_rotation(np.array([0, 0, 1]), cube.rotation_angle_z)

        # Upwards
        if g_keys[key.SPACE]:
            cube.position[1] -= speed
        # Downwards movement
        if g_keys[key.LCTRL] or g_keys[key.RCTRL]:
            cube.position[1] += speed

        # Normal movement
        if g_keys[key.D]:
            cube.position[0] -= speed
        if g_keys[key.A]:
            cube.position[0] += speed
        if g_keys[key.W]:
            cube.position[2] -= speed
        if g_keys[key.S]:
            cube.position[2] += speed

        # To change the FOV freely
        if g_keys[key.F3]:
            cube.fov += 5 * dt
        if g_keys[key.F4]:
            cube.fov -= 5 * dt

        with measure_duration("Prepare duration: ", show_duration):
            # Prepare for a new transformation
            cube.prepare()

        with measure_duration("Rotate duration: ", show_duration):
            # Rotate all the quaternions
            cube.rotate(qx * qy * qz)

        with measure_duration("Translate duration: ", show_duration):
            # Translate the relative postition
            cube.translate(cube.position)

        with measure_duration("Build triangles duration: ", show_duration):
            # Building the needed triangles to the screen
            cube.build_triangles(
                window_width=g_window_width, window_height=g_window_height
            )
        if show_triangle_count:
            triangle_count_label.text = f"Triangle Count: {cube.next_triangle_index}"

    def on_key_press(symbol, modifiers) -> None:  # 10 points
        # Turn on wireframe (show all triangles of the cube)
        if symbol == key.F1:
            cube.wireframe = not cube.wireframe

        # Turn on different colors for the clipping of triangles
        if symbol == key.F2:
            cube.clip_debug = not cube.clip_debug

    g_window.push_handlers(on_key_press)
    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
