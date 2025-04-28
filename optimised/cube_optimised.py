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

from contextlib import contextmanager
from datetime import datetime

from general_optimised import (
    distance_point_to_plane,
    line_intersect_plane,
    # Trigon,
    vector_normalize,
    # array_distance_point_to_plane,
    array_distance_to_plane,
)
import numpy as np
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternion_optimised import Quaternion

# Global variables
g_window_width = 1920
g_window_height = 1080
g_window = Window(g_window_width, g_window_height)

g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)

RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
MAGENTA = [255, 0, 255]
YELLOW = [255, 255, 0]
CYAN = [0, 255, 255]
FRAME_TIME: float = 1 / 60

# Debug colors
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


# Defines the vertices of a cube
g_vertices = 100 * np.transpose(
    np.array(
        [
            [-1, 1, -1, 1, -1, 1, -1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )
)

# Defines the triangles of a cube
g_triangle_indices = np.array(
    [
        [0, 1, 2],
        [1, 2, 3],
        [0, 1, 5],
        [0, 4, 5],
        [1, 5, 7],
        [1, 3, 7],
        [0, 2, 6],
        [0, 4, 6],
        [4, 5, 7],
        [4, 6, 7],
        [2, 3, 7],
        [2, 6, 7],
    ]
)

# Defines the colors of the cube triangles
g_colors = np.array(
    [RED, RED, GREEN, GREEN, BLUE, BLUE, MAGENTA, MAGENTA, YELLOW, YELLOW, CYAN, CYAN],
    dtype=np.uint32,
)


class Model_3d:
    vertices: np.ndarray
    "Array of vertex arrays"
    temporary_vertices: np.ndarray
    "Temporary copy of vertices for transformation"
    triangle_indices: np.ndarray
    "Array of triangle arrays"
    colors: np.ndarray
    "Array of color arrays"
    fov: float = 50
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

    def __init__(
        self, vertices: np.ndarray, triangle_indices: np.ndarray, colors: np.ndarray
    ) -> None:
        """
        Class that contains a 3D model that can be rotated, translated and projected.
        They can then be drawn to the screen using pyglet library.

        :param vertices: This is the vertices of the model.
        :param triangle_indices: This is the triangles of the model, referencing vertices.
        :param colors: This is the colors for the triangles, same size as triangle_indices.
        """
        self.vertices = vertices
        self.temporary_vertices = np.array([])
        self.triangle_indices = triangle_indices
        self.colors = colors
        self.batch = Batch()
        self.triangles = []
        self.lines = []
        self.position = np.array([0, 0, 200])
        self.next_triangle_index = 0

    def prepare(self) -> None:
        """
        Preparing for a new loop.
        Initializing temporary properties.
        """
        # Saving the old coordianates of vertices
        self.temporary_vertices = self.vertices.copy()

        # # Deleting the triangle objects in the list
        # for triangle in self.triangles:
        #     triangle.delete()
        # # Reset the triangle list
        # self.triangles.clear()

        for triangle in self.triangles:
            triangle.visible = False
        # Keep track of the next triangle
        self.next_triangle_index = 0

        # Deleting the line objects in the list
        for line in self.lines:
            line.delete()
        # Reset the line list
        self.lines.clear()

    def add_vertex(self, vertex: np.ndarray) -> None:
        """
        Add a new vertex.

        :param vertex: Vertex with x,y,z coordinate.
        """
        self.vertices = np.vstack((self.vertices, np.array([vertex])))

    def delete_vertex(self, index: int) -> None:
        """
        Delete vertex at given index, will also delete all triangles that uses the vertex.

        :param index: Index of the vertex.
        """
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

    def add_triangle(self, triangle: np.ndarray, color: np.ndarray) -> None:
        """
        Add triangle with a color.

        :param triangle: Index of vertices of the triangle.
        :param color: Color of the triangle.
        """
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))
        self.colors = np.vstack((self.colors, np.array([color])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        """
        Delete a triangle at given index.

        :param index: Index of the triangle.
        """
        self.triangle_indices = np.delete(self.triangle_indices, index, axis=0)
        self.colors = np.delete(self.colors, index, axis=0)

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotate all the vertices by a quaternion.

        :param quaternion: Quaternion object
        """
        self.temporary_vertices = quaternion.rotate_vectors(self.temporary_vertices)

    def rotate2(self, quaternion: Quaternion) -> None:
        """
        Rotate all the vertices by a quaternion.

        :param quaternion: Quaternion object
        """
        quaternion.rotate_vectors2(self.temporary_vertices)

    def translate(self, vector: np.ndarray) -> None:
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
            self.lines.append(
                Line(vertex1[0], vertex1[1], vertex2[0], vertex2[1], batch=self.batch)
            )
            self.lines.append(
                Line(vertex2[0], vertex2[1], vertex3[0], vertex3[1], batch=self.batch)
            )
            self.lines.append(
                Line(vertex1[0], vertex1[1], vertex3[0], vertex3[1], batch=self.batch)
            )

        # self.triangles.append(triangle)

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


duration_bool = False


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


def main():
    # Defining the cube object
    cube = Model_3d(g_vertices, g_triangle_indices, g_colors)

    # Example for deleting
    # cube.delete_vertex(0)
    # cube.delete_triangle(0)

    # Example of adding
    # cube.add_vertex(np.array([200, -200, 200]))
    # cube.add_triangle(np.array([8, 7, 6]), np.array(R))

    # Initial preperation incase the drawing function is called
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
        with measure_duration("Optimised rotate duration: ", duration_bool):
            cube.rotate(qx * qy * qz)
        with measure_duration("Old rotate duration: ", duration_bool):
            cube.rotate2(qx * qy * qz)

        # Translate the relative postition
        cube.translate(cube.position)

        # Building the needed triangles to the screen
        cube.build_triangles(window_width=g_window_width, window_height=g_window_height)

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
