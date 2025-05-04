from global_functions import (
    array_triangle_clip_against_plane,
    array_project_screen,
)
import numpy as np
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternions import Quaternion
from rgb_converting import LMS_to_RGB
from midpoint import midpoint_2d
from ray_tracing import form_factors_raytrace
import perlin
from pyglet.gl import GLubyte

g_window_width = 800
g_window_height = 600
g_window = Window(g_window_width, g_window_height)
g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)
FRAME_TIME: float = 1 / 60
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


class PerlinNoise:
    def __init__(
        self, width: int, height: int, grid_size: tuple[int, int] | None = None
    ):
        self.width = width
        self.height = height
        self.noise = perlin.noise(self.width, self.height, grid_size)

    def to_sprite(self, alpha: int = 50):
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
        return pyglet.sprite.Sprite(img=img)


class Ray_tracing_model_3d:
    fov: float = 100
    "Field of view"
    far: float = 1000
    "Far plane for camera"
    near: float = 0.1
    "Near plane for camera"
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

    def __init__(
        self,
        vertices: np.ndarray,
        triangle_indices: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:
        self.vertices = vertices
        self.temporary_vertices = np.array([])
        self.triangle_indices = triangle_indices
        self.colors = np.zeros((len(triangle_indices), 3), dtype=np.int32)
        self.batch = Batch()
        self.triangles = []
        self.next_triangle_index = 0
        self.next_line_index = 0
        self.lines = []
        self.position = np.array([0, 0, 0])
        self.emit_array = emit
        self.reflect_array = reflect
        self.normal_array: np.ndarray = np.array([])
        self.area_array: np.ndarray = np.array([])
        self.form_factor_array: np.ndarray = np.array([])

    def prepare(self) -> None:
        self.temporary_vertices = self.vertices.copy()

        # Dont show the triangles
        for triangle in self.triangles:
            triangle.visible = False
        # Keep track of the next triangle
        self.next_triangle_index = 0

    def add_vertex(self, vertex: np.ndarray) -> None:  # 5 points
        self.vertices = np.vstack((self.vertices, np.array([vertex])))

    def delete_vertex(self, index: int) -> None:  # 5 points
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
                self.triangle_indices[index_triangle - deleted_count] = triangle
        self.vertices = np.delete(self.vertices, index, axis=0)

    def add_triangle(
        self,
        triangle: np.ndarray,
        emit: np.ndarray,
        reflect: np.ndarray,
    ) -> None:  # 5 points
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))
        self.emit_array = np.vstack((self.emit_array, np.array([emit])))
        self.reflect_array = np.vstack((self.reflect_array, np.array([reflect])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        self.triangle_indices = np.delete(self.triangle_indices, index, axis=0)
        self.emit_array = np.delete(self.emit_array, index, axis=0)
        self.reflect_array = np.delete(self.reflect_array, index, axis=0)

    def rotate(self, quaternion: Quaternion) -> None:
        # up to 25 points, including the quaternion code
        self.temporary_vertices = quaternion.rotate_vectors(self.temporary_vertices)

    def translate(self, vector: np.ndarray) -> None:  # 10 points
        if vector.ndim != 1:
            raise ValueError("Vector must be 1D")
        if len(vector) != 3:
            raise ValueError("Vector length must be 3")

        for i in range(len(self.temporary_vertices)):
            self.temporary_vertices[i] = self.temporary_vertices[i] + vector

    def array_project(
        self, vertices: np.ndarray, window_width: int, window_height: int
    ) -> np.ndarray:
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

    def build_triangle(
        self,
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        vertex3: np.ndarray,
        color: np.ndarray,
        vis_check=False,
    ) -> None:

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

    def build_triangles(
        self, window_width: int, window_height: int, vis_check=False
    ) -> None:
        new_indices, new_vertices, new_colors = array_triangle_clip_against_plane(
            plane_p=np.array([0, 0, 0.1]),
            plane_n=np.array([0, 0, 1]),
            vertices=self.temporary_vertices,
            triangles=self.triangle_indices,
            colors=self.colors,
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
        midpoint: bool = False,
        midpoint_scale: float = 1.0,
        terrain_roughness: float = 1.0,
    ) -> "Ray_tracing_model_3d":
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
            grid_n = int(np.ceil(np.log2(size - 1)))
            square_size = 2**grid_n + 1
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
        num_triangles = len(self.triangle_indices)
        self.normal_array = np.zeros((num_triangles, 3))
        self.area_array = np.zeros(num_triangles)
        for i in range(num_triangles):
            index_vertex = self.triangle_indices[i]
            vertex_1 = self.vertices[index_vertex[0]]
            vertex_2 = self.vertices[index_vertex[1]]
            vertex_3 = self.vertices[index_vertex[2]]
            vector_1 = vertex_2 - vertex_1
            vector_2 = vertex_3 - vertex_1
            normal = np.cross(vector_1, vector_2)
            normalize = np.linalg.norm(normal)
            if normalize > 0:
                normal = normal / normalize
            area = normalize / 2
            self.normal_array[i] = normal
            self.area_array[i] = area

    def update_form_factors(
        self, raytrace: bool = False, n_rays: int = 10
    ) -> None:  # 5 points
        if raytrace:
            self.form_factor_array = form_factors_raytrace(
                self.vertices,
                self.triangle_indices,
                n_rays,
            )

    def find_luminances(self, num_iterations: int) -> np.ndarray:
        radiosity_array = self.emit_array.copy()
        initial_emission_array = self.emit_array.copy()
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
            reflected_light = reflected_light * mask_non_diagonal_expanded
            total_reflected = np.sum(reflected_light, axis=1)
            radiosity_array += total_reflected
        luminances = np.clip(radiosity_array, 0, 1)
        return luminances

    def find_colors(self, num_iterations: int) -> np.ndarray:  # 10 points
        lms_luminances = self.find_luminances(num_iterations)
        num_triangles = len(self.triangle_indices)
        rgb_colors = np.zeros((num_triangles, 3), dtype=np.int32)
        for index in range(num_triangles):
            rgb_colors[index] = LMS_to_RGB(lms_luminances[index])
        return rgb_colors

    def add_model(self, other: "Ray_tracing_model_3d") -> None:  # 10 points
        vertex_offset = len(self.vertices)
        self.vertices = np.vstack((self.vertices, other.vertices))
        offset_indices = other.triangle_indices + vertex_offset
        self.triangle_indices = np.vstack((self.triangle_indices, offset_indices))
        self.emit_array = np.vstack((self.emit_array, other.emit_array))
        self.reflect_array = np.vstack((self.reflect_array, other.reflect_array))


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
        midpoint=True,
        midpoint_scale=15.0,
        terrain_roughness=4.5,
    )
    cube.add_model(bottom_wall)

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
        emission=np.array([10.0, 10.0, 10.0]),  # Bright white light
        reflect=np.array([0.0, 0.0, 0.0]),
        translation_vector=np.array(
            [0, cube_size / 2 - 5, 0]
        ),  # Slightly below ceiling
        rotation_axis=np.array([1, 0, 0]),
        rotation_angle=np.pi / 2,  # Pointing down
    )
    cube.add_model(center_light)
    cube.update_geometry()
    cube.update_form_factors(raytrace=True)
    cube.colors = cube.find_colors(num_iterations=25)

    noise = PerlinNoise(
        width=g_window_width,
        height=g_window_height,
    )

    sprite = noise.to_sprite()
    cube.prepare()

    # Draw to the screen
    @g_window.event
    def on_draw() -> None:
        g_window.clear()
        sprite.draw()
        cube.batch.draw()

    def update(dt: float) -> None:
        # Building the needed triangles to the screen
        cube.build_triangles(window_width=g_window_width, window_height=g_window_height)

    pyglet.clock.schedule_interval(update, FRAME_TIME)
    pyglet.app.run()


if __name__ == "__main__":
    main()
