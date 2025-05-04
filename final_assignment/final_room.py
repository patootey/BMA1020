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
    To change to show visability check: V

@Author Mathilde Borsodi Hegna
"""

from contextlib import contextmanager

# from dataclasses import dataclass
from datetime import datetime

from pyglet.sprite import Sprite
from global_functions import (
    array_triangle_clip_against_plane,
    array_project_screen,
    rotate,
    translate,
)
from midpoint import midpoint_2d
from ray_tracing import form_factors_raytrace, get_visible_triangles
import numpy as np
import pyglet
from pyglet.graphics import Batch, Group
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternions import Quaternion
from rgb_converting import LMS_to_RGB
import perlin
from pyglet.gl import GLubyte

# Global variables
g_window_width = 800
g_window_height = 600
g_window = Window(g_window_width, g_window_height)

g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)

FRAME_TIME: float = 1 / 60


class PerlinNoise:
    def __init__(
        self, width: int, height: int, grid_size: tuple[int, int] | None = None
    ) -> None:
        """
        Class to handle noise and pyglet image conversion.

        :param width: The window width
        :param height: The window height
        :param grid_size: Tuple of x and y grid sizes.
        """
        self.width = width
        self.height = height
        self.noise = perlin.noise(self.width, self.height, grid_size)

    def to_sprite(self, alpha: int = 50) -> Sprite:
        """
        Convert noise to RGBA and to a bitmap so pyglet can read.

        :param alpha: The alpha channel value
        """
        color_channel = self.noise * 255

        # Generate an alpha channel
        alpha_channel = alpha * np.ones((self.height, self.width))

        # this makes the noise greyscale (sets RGB to be the same) and adds the alpha channel
        greyscale = np.dstack(
            [color_channel, color_channel, color_channel, alpha_channel]
        ).astype(np.uint32)

        greyscale = greyscale.reshape(self.height * self.width * 4)

        # convert the bitmap to a form pyglet can read
        byte_data = bytes((GLubyte * greyscale.size)(*greyscale.astype("uint8")))

        # create an image of the noise
        img = pyglet.image.ImageData(
            self.width, self.height, "RGBA", byte_data, pitch=self.width * 4
        )

        # set the noise into a sprite for displaying in the pyglet application
        return pyglet.sprite.Sprite(img=img)


class Ray_tracing_model_3d:
    vertices: np.ndarray
    "Array of vertex arrays"
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
    position: np.ndarray = np.array([0, 0, 0])
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
    visability_check: bool
    "Debugging tool to turn on visability"
    centroid_array: np.ndarray
    "Array of all centroids"
    next_line_index: int
    "Index of which line to be made"
    next_triangle_index: int
    "Index of which triangle to be made"
    triangle_group: Group
    "Triangles draw group"
    line_group: Group
    "Lines draw group"

    def __init__(
        self,
        vertices: np.ndarray,
        triangle_indices: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:
        # Previous named: nodes
        self.vertices = vertices
        # Previous named: elements
        self.triangle_indices = triangle_indices
        self.colors = np.zeros((len(triangle_indices), 3), dtype=np.int32)
        self.batch = Batch()
        # Changed from a dictionary to a list
        self.triangles = []
        self.lines = []
        self.emit_array = emit
        self.reflect_array = reflect
        self.normal_array: np.ndarray = np.array([])
        self.area_array: np.ndarray = np.array([])
        self.form_factor_array: np.ndarray = np.array([])
        self.visability_check = False
        self.centroid_array = np.array([])
        self.next_line_index = 0
        self.next_triangle_index = 0
        self.triangle_group = Group(0)
        self.line_group = Group(1)

    def prepare(self) -> None:
        """
        Preparing for a new loop.
        Initializing temporary properties.
        """
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
                group=self.triangle_group,
            )
            self.triangles.append(triangle)
            self.next_triangle_index += 1

        # Debugging helpers
        if self.wireframe:
            self.build_line(vertex1, vertex2)
            self.build_line(vertex2, vertex3)
            self.build_line(vertex1, vertex3)

    def build_line(
        self,
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        if self.next_line_index < len(self.lines):
            line = self.lines[self.next_line_index]
            line.x = vertex1[0]
            line.y = vertex1[1]
            line.x2 = vertex2[0]
            line.y2 = vertex2[1]
            line.color = color
            line.visible = True
            self.next_line_index += 1
        else:
            self.lines.append(
                Line(
                    vertex1[0],
                    vertex1[1],
                    vertex2[0],
                    vertex2[1],
                    color=color,
                    batch=self.batch,
                    group=self.line_group,
                )
            )
            self.next_line_index += 1

    def build_triangles(
        self,
        window_width: int,
        window_height: int,
        vertices: np.ndarray,
        centroids: np.ndarray,
        normals: np.ndarray,
        triangle_indices: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """
        Builds all the triangles for the viewport.
        The triangles will be rendered by pyglet.

        :param window_width: Window width.
        :param window_height: Window heigth.
        :param vertices: Array of vertices.
        :param centroids: Array of centroids.
        :param normals: Array of normals.
        :param triangle_indices: Array of triangle indices.
        :param colors: Array of colors.
        """
        if self.visability_check:
            visible_triangles = get_visible_triangles(
                camera_position=np.array([0, 0, -10]),
                triangle_indices=triangle_indices,
                vertices=vertices,
                centroids=centroids,
                normals=normals,
            )

            # Filter triangles using visibility check
            visible_indices = triangle_indices[visible_triangles]
            visible_colors = colors[visible_triangles]
        else:
            # Process all triangles if visibility check is disabled
            visible_indices = triangle_indices
            visible_colors = colors

        # Originally named: draw
        # Draw the model to the screen
        new_indices, new_vertices, new_colors = array_triangle_clip_against_plane(
            plane_p=np.array([0, 0, 0.1]),
            plane_n=np.array([0, 0, 1]),
            vertices=vertices,
            triangles=visible_indices,
            colors=visible_colors,
            clip_debug=self.clip_debug,
        )

        projected_vertices = array_project_screen(
            vertices=new_vertices,
            window_width=window_width,
            window_height=window_height,
            far=self.far,
            near=self.near,
            fov=self.fov,
        )

        # works, but has some problems with z values that are closest
        z_values = np.full(len(new_indices), np.nan)
        for triangle_index in range(len(new_indices)):
            triangle = new_indices[triangle_index]

            triangle1_z = new_vertices[triangle[0], 2]
            triangle2_z = new_vertices[triangle[1], 2]
            triangle3_z = new_vertices[triangle[2], 2]

            triangle_z = np.mean([triangle1_z, triangle2_z, triangle3_z])

            z_values[triangle_index] = triangle_z

        sorted_indices = np.argsort(z_values)[::-1]

        for triangle_index in sorted_indices:
            color = new_colors[triangle_index]
            triangle_array: np.ndarray = new_indices[triangle_index]
            vertex1_index, vertex2_index, vertex3_index = triangle_array

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
        midpoint: bool = False,
        midpoint_scale: float = 1.0,
        terrain_roughness: float = 1.0,
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
        :param midpoint: If true, midpoint/heightmap will be set.
        :param midpoint_scale: The scale for midpoint.
        :param terrain_roughness: How rough the heightmap should be.
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
        if midpoint:
            # Find the power of 2 that will fit our grid size
            size = max(x_divisions, y_divisions)
            # Calculate how many powers of 2 would be needed to represent this size
            grid_n = int(np.ceil(np.log2(size - 1)))

            square_size = 2**grid_n + 1

            # Create heightmap with the right dimensions
            heightmap = np.zeros((square_size + 1, square_size + 1))

            # Apply the midpoint displacement algorithm
            midpoint_2d(heightmap, 0, 0, grid_n, midpoint_scale, terrain_roughness)

            # Extract the portion we need based on our dimensions
            heightmap = heightmap[:y_divisions, :x_divisions]

            # Reshape heightmap to match z_values shape
            z_values = heightmap.flatten()[:, np.newaxis]

        vertices = np.hstack((flat_xy, z_values))
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

    def update_geometry(self) -> None:  # 5 points
        """
        Compute the normals, centroids and areas of each triangle in the plane.
        """
        num_triangles = len(self.triangle_indices)

        self.normal_array = np.zeros((num_triangles, 3))
        self.area_array = np.zeros(num_triangles)
        self.centroid_array = np.zeros((num_triangles, 3))

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
            centroid = np.mean(np.array([vertex_1, vertex_2, vertex_3]), axis=0)

            # Store results
            self.normal_array[i] = normal
            self.area_array[i] = area
            self.centroid_array[i] = centroid

    def update_form_factors(self, n_rays: int) -> None:  # 5 points
        """
        Compute the form factor ray trace for all triangle pairs.


        :param n_rays: Number of random rays to send out.
        """

        self.form_factor_array = form_factors_raytrace(
            vertices=self.vertices,
            triangles_indices=self.triangle_indices,
            # centroids=self.centroid_array,
            normals=self.normal_array,
            n_rays=n_rays,
        )

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
        non_diagonal = np.ones(
            (len(self.triangle_indices), len(self.triangle_indices)), dtype=bool
        )
        np.fill_diagonal(non_diagonal, False)

        for _ in range(num_iterations):
            prev_radiosity = radiosity_array.copy()

            radiosity_array = initial_emission_array.copy()

            prev_radiosity_expanded = prev_radiosity[np.newaxis, :, :]
            form_factors_expanded = self.form_factor_array[:, :, np.newaxis]
            reflect_expanded = self.reflect_array[:, np.newaxis, :]
            non_diagonal_expanded = non_diagonal[:, :, np.newaxis]

            light_contributions = form_factors_expanded * prev_radiosity_expanded
            reflected_light = reflect_expanded * light_contributions

            # only the ones that are not diagonal i == j
            reflected_light = reflected_light * non_diagonal_expanded

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
    cube_divisions = 8
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
        translation_vector=np.array([0, 0, cube_size / 2 + offset]),
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
        translation_vector=np.array([0, 0, -cube_size / 2]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi,
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
        translation_vector=np.array([cube_size / 2, 0, 0]),
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=-np.pi / 2,
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
        reflect=np.array([0.8, 0.5, 0.3]),
        translation_vector=np.array([-cube_size / 2, 0, 0]),  # Left position
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.pi / 2,
    )
    cube.add_model(left_wall)

    # Bottom wall, with heigthmap
    bottom_wall = cube.model_rect(
        -cube_size / 2,
        cube_size / 2,
        cube_divisions,
        -cube_size / 2,
        cube_size / 2 + offset,
        cube_divisions,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=np.array([0.5, 0.3, 0.0]),
        translation_vector=np.array([0, -cube_size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
        flip_normals=True,
        midpoint=True,
        midpoint_scale=15.0,
        terrain_roughness=4.5,
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
        reflect=np.array([0.2, 0.1, 0.2]),
        translation_vector=np.array([0, cube_size / 2, 0]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=-np.pi / 2,
        flip_normals=True,
    )
    cube.add_model(top_wall)

    # --------------------------------------------------------------------------

    column_size = 25
    column_height = cube_size - 5
    column_division = 5
    reflectance_gray = np.array([0.2, 0.2, 0.2])
    column_offset = cube_size / 2 - column_size
    base_position = np.array([column_offset, 0, column_offset])

    # Front
    column1_front = cube.model_rect(
        -column_size / 2,
        column_size / 2,
        column_division,
        -column_height / 2,
        column_height / 2,
        column_division,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=reflectance_gray,
        translation_vector=base_position + np.array([0, 0, -column_size / 2]),
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=0,
        flip_normals=True,
    )
    cube.add_model(column1_front)

    # Right
    column1_right = cube.model_rect(
        -column_size / 2,
        column_size / 2,
        column_division,
        -column_height / 2,
        column_height / 2,
        column_division,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=reflectance_gray,
        translation_vector=base_position + np.array([column_size / 2, 0, 0]),
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=-np.pi / 2,
        flip_normals=True,
    )
    cube.add_model(column1_right)

    # Back
    column1_back = cube.model_rect(
        -column_size / 2,
        column_size / 2,
        column_division,
        -column_height / 2,
        column_height / 2,
        column_division,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=reflectance_gray,
        translation_vector=base_position + np.array([0, 0, column_size / 2]),
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.pi,
        flip_normals=True,
    )
    cube.add_model(column1_back)

    # Left
    column1_left = cube.model_rect(
        -column_size / 2,
        column_size / 2,
        column_division,
        -column_height / 2,
        column_height / 2,
        column_division,
        emission=np.array([0.0, 0.0, 0.0]),
        reflect=reflectance_gray,
        translation_vector=base_position + np.array([-column_size / 2, 0, 0]),
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.pi / 2,
        flip_normals=True,
    )
    cube.add_model(column1_left)

    # --------------------------------------------------------------------------

    light_size = 30
    light_division = 4

    # Center ceiling light
    center_light = cube.model_rect(
        -light_size / 2,
        light_size / 2,
        light_division,
        -light_size / 2,
        light_size / 2,
        light_division,
        emission=np.array([100.0, 100.0, 100.0]),
        reflect=np.array([0.0, 0.0, 0.0]),
        translation_vector=np.array([0, cube_size / 2 - 2, 80]),
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,
    )
    cube.add_model(center_light)

    # Example for deleting
    # cube.delete_vertex(1)
    # cube.delete_triangle(5)

    # Example of adding
    # cube.add_vertex(np.array([200, -200, 200]))
    # cube.add_triangle(
    #     triangle=np.array([8, 7, 6]),
    #     emit=np.array([0, 0, 0]),
    #     reflect=np.array([0.3, 0.0, 0.2]),
    # )

    print("Update geometry...")
    cube.update_geometry()

    with measure_duration("Form factor duration: ", True):
        print("Calculating form factors...")
        cube.update_form_factors(n_rays=100)

    with measure_duration("Color duration: ", True):
        print("Calculating light luminances...")
        cube.colors = cube.find_colors(num_iterations=50)

    # Generate noise image
    noise = PerlinNoise(
        width=g_window_width,
        height=g_window_height,
    )
    sprite = noise.to_sprite(alpha=35)

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
        # Draw image to screen
        sprite.draw()

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
            (cube_vertices, cube_normals, cube_centroids) = rotate(
                qx * qy * qz,
                vertices=cube.vertices,
                normals=cube.normal_array,
                centroids=cube.centroid_array,
            )

        with measure_duration("Translate duration: ", show_duration):
            # Translate the relative postition
            (cube_vertices, cube_centroids) = translate(
                cube.position, vertices=cube_vertices, centroids=cube_centroids
            )

        with measure_duration("Build triangles duration: ", show_duration):
            # Building the needed triangles to the screen
            cube.build_triangles(
                window_width=g_window_width,
                window_height=g_window_height,
                vertices=cube_vertices,
                centroids=cube_centroids,
                normals=cube_normals,
                triangle_indices=cube.triangle_indices,
                colors=cube.colors,
                # vis_check=visability_check,
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

        if symbol == key.V:
            cube.visability_check = not cube.visability_check

    g_window.push_handlers(on_key_press)
    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
