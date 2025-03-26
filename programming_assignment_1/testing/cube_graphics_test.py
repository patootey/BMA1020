"""
This code generates a cube, rotates it and displays it on the screen.

The controls:
    To go upwards press SPACE
    To go downwards press CTRL
    Normal movement: WASD
    Turn the cube: arrow keys

Author: Mathilde Borsodi Hegna
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from programming_assignment_1.testing.quaternions_test import Quaternion
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window

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


# SORT FUNCTION: list.sort() se videoer
# sort before draw

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

# MARK: Vector functions


# General functions for vectors
def vector_dot_product(vector1: np.ndarray, vector2: np.ndarray):
    """
    Get the dot product of the vectors.
    """
    # Numpy does per element multiplication
    return np.sum(vector1 * vector2)


def vector_length(vec: np.ndarray):
    """
    Calculate length of the vector and return it.
    """
    return np.sqrt(vector_dot_product(vec, vec))


def vector_normalize(vec: np.ndarray):
    """
    Normalizing the vector.
    """
    length = vector_length(vec)
    return vec / length


def distance_point_to_plane(
    plane_n: np.ndarray, plane_p: np.ndarray, point: np.ndarray
) -> float:
    """
    Getting the distance between a vertex and plane.

    :param plane_n: The normal of the plane.
    :param plane_p: The position of the plane.
    :param point: The point position.
    """

    # Plane needs to be normalized
    plane_n = vector_normalize(plane_n)

    # Return shortes distance from point to plane
    return vector_dot_product(plane_n, point) - vector_dot_product(plane_n, plane_p)


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
    # Found this on the net, intersect plane with vector
    # Make sure the plane is normalized
    plane_n = vector_normalize(plane_n)

    # D elemnt of the plane (ABCD)
    plane_d = -vector_dot_product(plane_n, plane_p)
    ad = vector_dot_product(point_1, plane_n)
    bd = vector_dot_product(point_2, plane_n)

    # If line is parallel with the plane
    if ad == bd:
        return None

    t = (-plane_d - ad) / (bd - ad)

    vec = point_2 - point_1
    intersect = vec * t

    return point_1 + intersect


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
    triangles: list[pyglet.shapes.Triangle]
    "Store all pyglet triangles that are created"
    clip_debug: bool = False
    "Debugging tool to show when the triangles are clipped"
    wireframe: bool = False
    "Debugging tool to show all the triangles in the cube"
    lines: list[pyglet.shapes.Line]
    "Store all pyglet lines that are created to show a wireframe"
    rotation_angle_x: float = 0
    "The rotation angle of the cube around the X-axis"
    rotation_angle_y: float = 0
    "The rotation angle of the cube around the Y-axis"
    rotation_angle_z: float = 0
    "The rotation angle of the cube around the Z-axis"
    position: np.ndarray
    "The position of the cube"
    camera_position: np.ndarray
    "The position of the camera"
    camera_rotation_angle: float = 0
    "The rotation angle of the camera"

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
        # Previous named: nodes
        self.vertices = vertices
        self.temporary_vertices = np.array([])
        # Previous named: elements
        self.triangle_indices = triangle_indices
        self.colors = colors
        self.batch = Batch()
        # Changed from a dictionary to a list
        self.triangles = []
        self.lines = []
        self.position = np.array([0, 0, 200])
        self.camera_position = np.array([0, 0, 0])

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

    def add_triangle(self, triangle: np.ndarray, color: np.ndarray) -> None:  # 5 points
        """
        Add triangle with a color.

        :param triangle: Index of vertices of the triangle.
        :param color: Color of the triangle.
        """
        # Previously named: add_element
        # And removed back parameter
        # Add an triangle to the model
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))
        # remember that each element has an associated color...
        self.colors = np.vstack((self.colors, np.array([color])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        """
        Delete a triangle at given index.

        :param index: Index of the triangle.
        """
        # Previously named: delete_element
        # Remove an triangle from the model
        self.triangle_indices = np.delete(self.triangle_indices, index, axis=0)
        self.colors = np.delete(self.colors, index, axis=0)

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotate all the vertices by a quaternion.

        :param quaternion: Quaternion object
        """
        # up to 25 points, including the quaternion code
        # Rotate each vertex for the wanted roation with quaternions
        self.temporary_vertices = quaternion.rotate_vectors(self.temporary_vertices)

    def translate(self, vector: np.ndarray) -> None:  # 10 points
        """
        Translating all vertices by the vector.

        :param vector: Vector direction.
        """
        # Translate the vertices with a vector
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

        # Bulding the triangles shapes
        triangle = pyglet.shapes.Triangle(
            vertex1[0],
            vertex1[1],
            vertex2[0],
            vertex2[1],
            vertex3[0],
            vertex3[1],
            color=tuple(trigon.color),
            batch=self.batch,
        )

        # Debugging helpers
        if self.wireframe:
            self.lines.append(
                pyglet.shapes.Line(
                    vertex1[0], vertex1[1], vertex2[0], vertex2[1], batch=self.batch
                )
            )
            self.lines.append(
                pyglet.shapes.Line(
                    vertex2[0], vertex2[1], vertex3[0], vertex3[1], batch=self.batch
                )
            )
            self.lines.append(
                pyglet.shapes.Line(
                    vertex1[0], vertex1[1], vertex3[0], vertex3[1], batch=self.batch
                )
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
            color = self.colors[triangle_index]
            triangle_array: np.ndarray = self.triangle_indices[triangle_index]
            vertex1_index, vertex2_index, vertex3_index = triangle_array
            # Make sure the vertices does not get the same coords/index? noe i den duren?
            vertex1 = self.temporary_vertices[vertex1_index].copy()
            vertex2 = self.temporary_vertices[vertex2_index].copy()
            vertex3 = self.temporary_vertices[vertex3_index].copy()

            # CLIP
            trigons = self.triangle_clip_against_plane(
                np.array([0, 0, 0.1]), np.array([0, 0, 1]), vertex1, vertex2, vertex3
            )

            # GET NEW TRIANGLES
            for trigon in trigons:
                if len(trigon.color) == 0:
                    trigon.color = color

                trigon.index = triangle_index

                self.build_triangle(trigon, window_width, window_height)


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

        if g_keys[key.Q]:
            cube.camera_rotation_angle += rotation_speed
        if g_keys[key.E]:
            cube.camera_rotation_angle -= rotation_speed

        camera_orientation = Quaternion.from_rotation(
            np.array([0, 1, 0]), cube.camera_rotation_angle
        )

        if g_keys[key.LEFT]:
            cube.rotation_angle_y += rotation_speed
        if g_keys[key.RIGHT]:
            cube.rotation_angle_y -= rotation_speed
        if g_keys[key.UP]:
            cube.rotation_angle_x += rotation_speed
        if g_keys[key.DOWN]:
            cube.rotation_angle_x -= rotation_speed

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

        # Rotate the cube for the x-axis
        view_quat = camera_orientation
        cube.rotate(view_quat)

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
