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

Author: Mathilde Borsodi Hegna
"""

from global_functions2 import (
    distance_point_to_plane,
    line_intersect_plane,
    Trigon,
    vector_normalize,
    LMS_to_RGB,
)
import numpy as np
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternions2 import Quaternion

# Global variables
g_window_width = 1920
g_window_height = 1080
g_window = Window(g_window_width, g_window_height)

g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)

# RED = [255, 0, 0]
# GREEN = [0, 255, 0]
# BLUE = [0, 0, 255]
# MAGENTA = [255, 0, 255]
# YELLOW = [255, 255, 0]
# CYAN = [0, 255, 255]
FRAME_TIME: float = 1 / 60

# Debug colors
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


class Light_room:
    vertices: np.ndarray
    "Array of vertex arrays"
    temporary_vertices: np.ndarray
    "Temporary copy of vertices for transformation"
    triangle_indices: np.ndarray
    "Array of triangle arrays"
    # colors: np.ndarray
    # "Array of color arrays"
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
    emit: np.ndarray
    "Array of emission values for triangles"
    reflect: np.ndarray
    "Array of reflection values for triangles"

    # CHANGE OUT THE COLOR WITH EMIT AND REFLECT
    def __init__(
        self,
        vertices: np.ndarray,
        triangle_indices: np.ndarray,
        # colors: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:
        # Previous named: nodes
        self.vertices = vertices
        self.temporary_vertices = np.array([])
        # Previous named: elements
        self.triangle_indices = triangle_indices
        # self.colors = colors
        self.batch = Batch()
        # Changed from a dictionary to a list
        self.triangles = []
        self.lines = []
        self.position = np.array([0, 0, 200])

        # THE NEW
        self.emit = emit
        self.reflect = reflect

        self.normals: np.ndarray = np.array([])
        self.areas: np.ndarray = np.array([])

        self.form_factors: np.ndarray = np.array([])

    def prepare(self) -> None:
        """
        Preparing for a new loop.
        Initializing temporary properties.
        """
        # Saving the old coordianates of vertices
        self.temporary_vertices = self.vertices.copy()

        # Deleting the triangle objects in the list
        for triangle in self.triangles:
            triangle.delete()
        # Reset the triangle list
        self.triangles.clear()

        # Deleting the line objects in the list
        for line in self.lines:
            line.delete()
        # Reset the line list
        self.lines.clear()

    # FIX ALL ADD AND DELETE (for the new color)
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
        # color: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:  # 5 points
        """
        Add triangle with a color.

        :param triangle: Index of vertices of the triangle.
        :param color: Color of the triangle.
        """
        # Previously named: add_element
        # And removed back parameter
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))

        # NEW
        self.emit = np.vstack((self.emit, np.array([emit])))
        self.reflect = np.vstack((self.reflect, np.array([reflect])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        """
        Delete a triangle at given index.

        :param index: Index of the triangle.
        """
        # Previously named: delete_element
        self.triangle_indices = np.delete(self.triangle_indices, index, axis=0)

        self.emit = np.delete(self.emit, index, axis=0)
        self.reflect = np.delete(self.reflect, index, axis=0)

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

        if trigon.index >= 0:
            # ER DETTE RIKTIG???
            if self.luminances is not None:
                lms_color = self.luminances[trigon.index]
            else:
                lms_color = self.emit[trigon.index]
                # ANDERLEDES??
                # CAROLINE DID DIFF, jeg hardkoder tydelgivis
            # Convert LMS to RGB and scale to 0-255
            rgb_color = LMS_to_RGB(lms_color)
            display_color = (rgb_color * 255).astype(np.uint8)
        else:
            display_color = trigon.color

        # Bulding the triangles shapes
        triangle = Triangle(
            vertex1[0],
            vertex1[1],
            vertex2[0],
            vertex2[1],
            vertex3[0],
            vertex3[1],
            color=tuple(display_color),
            batch=self.batch,
        )

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

        self.triangles.append(triangle)

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
        # Draw the model to the screen
        # Rebuild the triangles for each edge
        for triangle_index in range(len(self.triangle_indices)):
            # color = self.colors[triangle_index]
            triangle_array: np.ndarray = self.triangle_indices[triangle_index]
            vertex1_index, vertex2_index, vertex3_index = triangle_array
            # Make sure the vertices does not get the same coords/index? noe i den duren?
            vertex1 = self.temporary_vertices[vertex1_index].copy()
            vertex2 = self.temporary_vertices[vertex2_index].copy()
            vertex3 = self.temporary_vertices[vertex3_index].copy()

            # Clipping the triangles
            trigons = self.triangle_clip_against_plane(
                np.array([0, 0, 0.1]), np.array([0, 0, 1]), vertex1, vertex2, vertex3
            )

            # Get the new triangles
            for trigon in trigons:

                trigon.index = triangle_index

                self.build_triangle(trigon, window_width, window_height)

    def add_model(self, other):  # 10 points
        # Store the current number of vertices to offset the indices from the other model
        vertex_offset = len(self.vertices)

        # Append vertices
        self.vertices = np.vstack((self.vertices, other.vertices))

        # Offset triangle indices and append
        offset_indices = other.triangle_indices + vertex_offset
        self.triangle_indices = np.vstack((self.triangle_indices, offset_indices))

        # Append emit and reflect values
        self.emit = np.vstack((self.emit, other.emit))
        self.reflect = np.vstack((self.reflect, other.reflect))

    def update_geometry(self):  # 5 points
        """
        Compute the normals and areas of each triangle in the plane
        """
        num_triangles = len(self.triangle_indices)

        self.normals = np.zeros((num_triangles, 3))
        self.areas = np.zeros(num_triangles)

        for i in range(num_triangles):
            # Get indices of the three vertices
            idx_vertex = self.triangle_indices[i]

            # Get the vertices
            vertex_1 = self.vertices[idx_vertex[0]]
            vertex_2 = self.vertices[idx_vertex[1]]
            vertex_3 = self.vertices[idx_vertex[2]]

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
            self.normals[i] = normal
            self.areas[i] = area

    def update_form_factors(self):  # 5 points
        """
        Compute the form factor matrix.
        Uses the formula:
            Fij = (1/Ai) * ∫∫ (cos(θi)cos(θj)/(π*r²)) dAidAj
        """
        # JVOR SKAL DISSE???
        self.update_geometry()

        num_triangles = len(self.triangle_indices)
        self.form_factors = np.zeros((num_triangles, num_triangles))

        for i in range(num_triangles):
            for j in range(num_triangles):
                if i == j:
                    # No self-illumination
                    self.form_factors[i, j] = 0.0
                    continue

                # Get the coordinates of the nodes for elements i and j

                centroid_i = np.mean(self.vertices[self.triangle_indices[i]], axis=0)
                centroid_j = np.mean(self.vertices[self.triangle_indices[j]], axis=0)

                # Vector from i to j
                r_vector = centroid_j - centroid_i
                r_normalize = np.linalg.norm(r_vector)

                if r_normalize == 0:
                    self.form_factors[i, j] = 0.0
                    continue

                # Unit direction vector
                r_unit = r_vector / r_normalize

                dot_ni = np.dot(r_unit, self.normals[i])
                dot_nj = np.dot(-r_unit, self.normals[j])

                if dot_ni <= 0 or dot_nj <= 0:
                    self.form_factors[i, j] = 0.0
                    continue

                F_ij = self.areas[j] * (dot_ni * dot_nj) / (np.pi * r_normalize**2)

                self.form_factors[i, j] = F_ij

        return self.form_factors

    def find_luminances(self, num_iterations=100):
        # if self.form_factors is None:
        self.update_form_factors()

        num_triangles = len(self.triangle_indices)

        # Initialize radiosity with just emission (no ambient light)
        radiosity = self.emit.copy()

        # Store a copy of the initial emission for convergence
        initial_emission = self.emit.copy()

        for _ in range(num_iterations):
            prev_radiosity = radiosity.copy()
            for i in range(num_triangles):
                # Start with direct emission only (no ambient term)
                radiosity[i] = initial_emission[i].copy()

                # Add reflected component from all other surfaces
                for j in range(num_triangles):
                    if i == j:
                        continue
                    # Light that bounces from surface j to surface i
                    radiosity[i] += (
                        self.reflect[i] * self.form_factors[i, j] * prev_radiosity[j]
                    )

        # Store and return results
        self.luminances = np.clip(radiosity, 0, 1)
        return self.luminances

    def find_colors(self, num_iterations=100):  # 10 points
        """
        Calculate RGB colors from LMS luminances.

        Args:
            num_iterations: Number of iterations for the radiosity calculation

        Returns:
            Array of RGB colors for each triangle
        """
        # Calculate LMS luminances
        lms_luminances = self.find_luminances(num_iterations)

        # Initialize RGB colors array
        num_triangles = len(self.triangle_indices)
        rgb_colors = np.zeros((num_triangles, 3))

        # Convert LMS to RGB for each triangle
        for index in range(num_triangles):
            rgb_colors[index] = LMS_to_RGB(lms_luminances[index])

        # # Store the results
        # self.colors = rgb_colors

        return rgb_colors

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
        flip_normals: bool = False,
    ):
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

        for x_div in range(num_y_divisions - 1):
            for y_div in range(num_x_divisions - 1):
                # The grid is laid out row by row
                top_left = x_div * num_x_divisions + y_div
                top_right = top_left + 1
                bottom_left = (x_div + 1) * num_x_divisions + y_div
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

        # Apply rotation to all vertices at once using rotate_vectors
        rotated_vectors = quaternion_rotation.rotate_vectors(vertices)

        # Apply translation to all rotated vertices
        for index in range(len(vertices)):
            vertices[index] = rotated_vectors[index] + np.array(translated_vector)

        # Create arrays for emission and reflection values for each triangle
        emit_array = np.array([emission_color for _ in range(len(triangle_indices))])
        reflect_array = np.array([reflect for _ in range(len(triangle_indices))])

        # Return all components needed to create the rectangle model
        return vertices, triangle_indices, emit_array, reflect_array


def main():
    cube = Light_room(
        vertices=np.empty((0, 3)),
        triangle_indices=np.empty((0, 3), dtype=int),
        emit=np.empty((0, 3)),
        reflect=np.empty((0, 3)),
    )

    size = 100

    cube_divisions = 5

    # Front face - Red
    front_vertices, front_indices, front_emit, front_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.2, 0.0, 0.0]),  # Red emission
        reflect=np.array([0.3, 0.1, 0.2]),
        translated_vector=np.array([0, 0, size / 2]),  # Front position
        rotation_axis=np.array([0, 0, 1]),
        rotation_angle=0,  # No rotation needed
        flip_normals=True,
    )

    front_face = Light_room(
        vertices=front_vertices,
        triangle_indices=front_indices,
        emit=front_emit,
        reflect=front_reflect,
    )
    cube.add_model(front_face)

    # Back face - Cyan
    back_vertices, back_indices, back_emit, back_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.0, 0.2, 0.2]),  # Cyan emission
        reflect=np.array([0.2, 0.3, 0.4]),
        translated_vector=np.array([0, 0, -size / 2]),  # Back position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi,  # 180 degree rotation
        flip_normals=True,
    )
    back_face = Light_room(
        vertices=back_vertices,
        triangle_indices=back_indices,
        emit=back_emit,
        reflect=back_reflect,
    )
    cube.add_model(back_face)

    # Right face - Pink
    right_vertices, right_indices, right_emit, right_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.5, 0.1, 0.2]),  # Pink emission
        reflect=np.array([0.3, 0.5, 0.5]),
        translated_vector=np.array([size / 2, 0, 0]),  # Right position
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=-np.pi / 2,  # -90 degree rotation
    )
    right_face = Light_room(
        vertices=right_vertices,
        triangle_indices=right_indices,
        emit=right_emit,
        reflect=right_reflect,
    )
    cube.add_model(right_face)

    # Left face - Purple
    left_vertices, left_indices, left_emit, left_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.1, 0.0, 0.2]),  # Purple emission
        reflect=np.array([0.2, 0.5, 0.5]),
        translated_vector=np.array([-size / 2, 0, 0]),  # Left position
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.pi / 2,  # 90 degree rotation
    )
    left_face = Light_room(
        vertices=left_vertices,
        triangle_indices=left_indices,
        emit=left_emit,
        reflect=left_reflect,
    )
    cube.add_model(left_face)

    # Top face - Green with two light sources
    top_vertices, top_indices, top_emit, top_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.0, 0.2, 0.0]),  # Green emission
        reflect=np.array([0.3, 0.5, 0.5]),
        translated_vector=np.array([0, -size / 2, 0]),  # Top position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,  # 90 degree rotation
        flip_normals=True,
    )
    top_face = Light_room(
        vertices=top_vertices,
        triangle_indices=top_indices,
        emit=top_emit,
        reflect=top_reflect,
    )
    cube.add_model(top_face)
    # Bottom face - Yellow
    bottom_vertices, bottom_indices, bottom_emit, bottom_reflect = cube.model_rect(
        -size / 2,
        size / 2,
        cube_divisions,
        -size / 2,
        size / 2,
        cube_divisions,
        emission_color=np.array([0.2, 0.2, 0.0]),  # Yellow emission
        reflect=np.array([0.3, 0.5, 0.5]),
        translated_vector=np.array([0, size / 2, 0]),  # Bottom position
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=-np.pi / 2,  # -90 degree rotation
        flip_normals=True,
    )

    bottom_face = Light_room(
        vertices=bottom_vertices,
        triangle_indices=bottom_indices,
        emit=bottom_emit,
        reflect=bottom_reflect,
    )
    cube.add_model(bottom_face)

    light_size = 30
    light_division = 2

    light1_vertices, light1_indices, light1_emit, light1_reflect = cube.model_rect(
        -size / 5 - light_size / 2,
        -size / 5 + light_size / 2,
        light_division,
        -size / 5 - light_size / 2,
        -size / 5 + light_size / 2,
        light_division,
        emission_color=np.array([2.0, 2.0, 2.0]),  # Strong white emission
        reflect=np.array([0.0, 0.0, 0.0]),
        translated_vector=np.array([0, size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
    )

    top_light = Light_room(
        vertices=light1_vertices,
        triangle_indices=light1_indices,
        emit=light1_emit,
        reflect=light1_reflect,
    )
    cube.add_model(top_light)

    light2_vertices, light2_indices, light2_emit, light2_reflect = cube.model_rect(
        size / 5 - light_size / 2,
        size / 5 + light_size / 2,
        light_division,
        size / 5 - light_size / 2,
        size / 5 + light_size / 2,
        light_division,
        emission_color=np.array([2.0, 2.0, 2.0]),  # Strong white emission
        reflect=np.array([0.0, 0.0, 0.0]),
        translated_vector=np.array([0, size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
    )

    top2_light = Light_room(
        vertices=light2_vertices,
        triangle_indices=light2_indices,
        emit=light2_emit,
        reflect=light2_reflect,
    )
    cube.add_model(top2_light)

    print("Calculating form factors...")
    cube.update_form_factors()

    print("Calculating light luminaces...")
    cube.find_luminances(num_iterations=50)

    # Initial preparation in case the drawing function is called
    cube.prepare()

    # Draw to the screen
    @g_window.event
    def on_draw() -> None:
        g_window.clear()
        cube.batch.draw()

    def update(dt) -> None:
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

        # Prepare for a new transformation
        cube.prepare()

        # Rotate all the quaternions
        cube.rotate(qx * qy * qz)

        # Translate the relative postition
        cube.translate(cube.position)

        # Building the needed triangles to the screen
        cube.build_triangles(
            window_width=g_window_height, window_height=g_window_height
        )

    def on_key_press(symbol, modifiers):  # 10 points
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
