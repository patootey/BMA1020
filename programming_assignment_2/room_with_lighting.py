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
    Trigon,
    vector_normalize,
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


class Light_model_3d:
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
        self.next_triangle_index = 0
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

        # Deleting the line objects in the list
        for line in self.lines:
            line.delete()
        # Reset the line list
        self.lines.clear()

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

    def project(
        self, vertex: np.ndarray, window_width: int, window_height: int
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
        # homogeneous coordinates
        org_x, org_y, org_z = vertex
        org_w = 1

        # Normalize to viewing volume
        new_x = aspectratio * f_rad * org_x
        new_y = f_rad * org_y
        new_z = q * org_z + (-self.near * q) * org_w
        new_w = org_w * org_z

        # Perspective transformation
        # Due to normalization and clipping, we know it will never divide by 0
        new_x = new_x / new_w
        new_y = new_y / new_w
        new_z = new_z / new_w
        # New_w disappears due to division by itself
        new_w = new_w / new_w

        projected_vertex = np.array([new_x, new_y, new_z])
        return projected_vertex

    def triangle_clip_against_plane(
        self,
        plane_p: np.ndarray,
        plane_n: np.ndarray,
        vertex_1: np.ndarray,
        vertex_2: np.ndarray,
        vertex_3: np.ndarray,
    ) -> list[Trigon]:
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
            return []

        if len(inside) == 3:
            # All the vertices are inside
            # Index will be filled in by the caller
            return [Trigon(vertex_1, vertex_2, vertex_3, -1, np.array([]))]

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
                return []

            return [
                Trigon(
                    inside[0],
                    inside[1],
                    intersect_vertex_1,
                    -1,
                    np.array(PINK if self.clip_debug else []),
                ),
                Trigon(
                    inside[1],
                    intersect_vertex_2,
                    intersect_vertex_1,
                    -1,
                    np.array(WHITE if self.clip_debug else []),
                ),
            ]

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
                return []

            return [
                Trigon(
                    inside[0],
                    intersect_vertex_2,
                    intersect_vertex_1,
                    -1,
                    np.array(GREY if self.clip_debug else []),
                )
            ]
        # This should never occur
        raise Exception("Something very wrong has happend")

    def build_triangle(
        self, trigon: Trigon, window_width: int, window_height: int
    ) -> None:  # 10 points
        """
        Building one triangle shape by projecting the vertex to the camera and transform to the viewport.
        Then adds the triangle to the active triangle list.
        If wireframe is on, it will draw lines on top of the triangle.

        :param trigon: The trigon object.
        :param window_width: Window width.
        :param window_heigth: Window heigth.
        """
        # Originally named: draw_index
        half_width = window_width / 2
        half_height = window_height / 2

        # Copying the vertex to avoid modifing the original values
        vertex1 = trigon.vertex_1.copy()
        vertex2 = trigon.vertex_2.copy()
        vertex3 = trigon.vertex_3.copy()

        # Projecting all the vertices to the camera plane
        vertex1 = self.project(vertex1, window_width, window_height)
        vertex2 = self.project(vertex2, window_width, window_height)
        vertex3 = self.project(vertex3, window_width, window_height)

        # Centering for viewport adjustments
        center = np.array([1, 1, 0])
        vertex1 += center
        vertex2 += center
        vertex3 += center

        # Scale into view
        scale = np.array([half_width, half_height, 1])
        vertex1 *= scale
        vertex2 *= scale
        vertex3 *= scale

        # Reuse the triangles already drawn if possible
        if self.next_triangle_index < len(self.triangles):
            triangle = self.triangles[self.next_triangle_index]
            triangle.x = vertex1[0]
            triangle.y = vertex1[1]
            triangle.x2 = vertex2[0]
            triangle.y2 = vertex2[1]
            triangle.x3 = vertex3[0]
            triangle.y3 = vertex3[1]
            triangle.color = tuple(trigon.color)
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
                color=tuple(trigon.color),
                batch=self.batch,
            )
            self.triangles.append(triangle)
            self.next_triangle_index += 1

        # Debugging helpers
        if self.wireframe:
            self.lines.append(
                Line(vertex1[0], vertex1[1], vertex2[0], vertex2[1], batch=self.batch)
            )
            self.lines.append(
                Line(vertex2[0], vertex2[1], vertex3[0], vertex3[1], batch=self.batch)
            )
            self.lines.append(
                Line(vertex1[0], vertex1[1], vertex3[0], vertex3[1], batch=self.batch)
            )

    def build_triangles(
        self, window_width: int, window_height: int
    ) -> None:  # up to 25 points
        """
        Builds all the triangles for the viewport.
        The triangles will be rendered by pyglet.

        :param window_width: Window width.
        :param window_height: Window heigth.
        """
        # Originally named: draw
        # Rebuild the triangles for each edge
        for triangle_index in range(len(self.triangle_indices)):
            color = self.colors[triangle_index]
            triangle_array: np.ndarray = self.triangle_indices[triangle_index]
            vertex1_index, vertex2_index, vertex3_index = triangle_array
            # Make sure the vertices does not get the same coords/index
            vertex1 = self.temporary_vertices[vertex1_index]
            vertex2 = self.temporary_vertices[vertex2_index]
            vertex3 = self.temporary_vertices[vertex3_index]

            # Clipping the triangles
            trigons = self.triangle_clip_against_plane(
                np.array([0, 0, 0.1]), np.array([0, 0, 1]), vertex1, vertex2, vertex3
            )

            # Get the new triangles
            for trigon in trigons:
                if len(trigon.color) == 0:
                    trigon.color = color
                trigon.index = triangle_index
                self.build_triangle(trigon, window_width, window_height)

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
    ) -> "Light_model_3d":
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

        return Light_model_3d(
            translated_vertices, triangle_indices, emit_array, reflect_array
        )

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

        for i in range(num_triangles):
            for j in range(num_triangles):
                if i == j:
                    # No self-illumination
                    self.form_factor_array[i, j] = 0.0
                    continue

                # Get the coordinates of the vertices for elements i and j
                centroid_i = np.mean(self.vertices[self.triangle_indices[i]], axis=0)
                centroid_j = np.mean(self.vertices[self.triangle_indices[j]], axis=0)

                # Vector from i to j
                r_vector = centroid_j - centroid_i
                r_normalize = np.linalg.norm(r_vector)

                if r_normalize == 0:
                    self.form_factor_array[i, j] = 0.0
                    continue

                # Unit direction vector
                r_unit = r_vector / r_normalize

                dot_ni = np.dot(r_unit, self.normal_array[i])
                dot_nj = np.dot(-r_unit, self.normal_array[j])

                if dot_ni <= 0 or dot_nj <= 0:
                    self.form_factor_array[i, j] = 0.0
                    continue

                F_ij = self.area_array[j] * (dot_ni * dot_nj) / (np.pi * r_normalize**2)

                self.form_factor_array[i, j] = F_ij

    def find_luminances(self, num_iterations: int = 50) -> np.ndarray:
        """
        Compute the final luminance (radiosity) values for each triangle
        using an iterative radiosity method.
        Used some from assignment 8.

        :param num_iterations: Number of iterations for the radiosity computation.
        :return: A array of RGB luminance values.
        """

        num_triangles = len(self.triangle_indices)

        # Initialize radiosity with just emission (no ambient light)
        radiosity_array = self.emit_array.copy()

        # Store a copy of the initial emission for convergence
        initial_emission_array = self.emit_array.copy()

        for _ in range(num_iterations):
            prev_radiosity = radiosity_array.copy()
            for i in range(num_triangles):
                # Start with direct emission only (no ambient term)
                radiosity_array[i] = initial_emission_array[i].copy()

                # Add reflected component from all other surfaces
                for j in range(num_triangles):
                    if i == j:
                        continue
                    # Light that bounces from surface j to surface i
                    radiosity_array[i] += (
                        self.reflect_array[i]
                        * self.form_factor_array[i, j]
                        * prev_radiosity[j]
                    )

        # Store and return results
        self.luminances = np.clip(radiosity_array, 0, 1)
        return self.luminances

    def find_colors(self, num_iterations: int = 50) -> np.ndarray:  # 10 points
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

    def add_model(self, other: "Light_model_3d") -> None:  # 10 points
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
    cube = Light_model_3d(
        vertices=np.empty((0, 3)),
        triangle_indices=np.empty((0, 3), dtype=int),
        emit=np.empty((0, 3)),
        reflect=np.empty((0, 3)),
    )

    cube_size = 100
    cube_divisions = 10

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
        translation_vector=np.array([0, 0, cube_size / 2]),  # Front position
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
        cube_size / 2,
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
        -cube_size / 2,
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
        cube_size / 2,
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
        -cube_size / 2,
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

    print("Update geometry...")
    cube.update_geometry()

    with measure_duration("Form factor duration: ", True):
        print("Calculating form factors...")
        cube.update_form_factors()

    with measure_duration("Color duration: ", True):
        print("Calculating light luminances...")
        cube.colors = cube.find_colors(num_iterations=10)

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
