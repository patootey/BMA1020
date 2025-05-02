from global_functions import (
    distance_point_to_plane,
    line_intersect_plane,
    vector_normalize,
    array_distance_to_plane,
    midpoint_2d,
    form_factors_raytrace,
)
import numpy as np
import pyglet
from pyglet.graphics import Batch
from pyglet.window import key, Window
from pyglet.shapes import Line, Triangle
from quaternions import Quaternion
from rgb_converting import LMS_to_RGB

# from pyglet.gl import GLubyte

# Global variables
g_window_width = 800
g_window_height = 600
g_window = Window(g_window_width, g_window_height)

g_keys = key.KeyStateHandler()
g_window.push_handlers(g_keys)

FRAME_TIME: float = 1 / 60
PINK = [255, 0, 128]
WHITE = [255, 255, 255]
GREY = [200, 200, 200]


def lerp(a, b, t):
    # Linear interpolation
    return a * (1 - t) + b * t


class PerlinNoise:
    def __init__(
        self,
        width: int = 400,
        height: int = 300,
        grid_x_size: int = 11,
        grid_y_size: int = 9,
        seed=12345,
    ):
        self.width = width
        self.height = height
        self.grid_x_size = grid_x_size
        self.grid_y_size = grid_y_size
        self.rng = np.random.default_rng(seed)

        self.cell_width = width // (grid_x_size - 1)
        self.cell_height = height // (grid_y_size - 1)

        # Create grid
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(0, width, grid_x_size), np.linspace(0, height, grid_y_size)
        )

        # Generate random gradient vectors
        self.angles = 2 * np.pi * self.rng.random((grid_x_size, grid_y_size))
        self.vectors = np.array([np.cos(self.angles), np.sin(self.angles)])

    def easeCurve(self, t):
        # Perlins ease curve
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise2(self, x: float, y: float):
        # Define i and j
        i = np.clip(int(x / self.cell_width), 0, self.grid_x_size - 2)
        j = np.clip(int(y / self.cell_height), 0, self.grid_y_size - 2)

        # Get corner coordinates
        # bl = bottom left, br = bottom right, tl = top left, tr = top right
        x_bl = self.x_grid[j, i]
        y_bl = self.y_grid[j, i]
        x_br = self.x_grid[j, i + 1]
        y_br = self.y_grid[j, i + 1]
        x_tl = self.x_grid[j + 1, i]
        y_tl = self.y_grid[j + 1, i]
        x_tr = self.x_grid[j + 1, i + 1]
        y_tr = self.y_grid[j + 1, i + 1]

        # Calculate relative position within cell
        tx = (x - x_bl) / (x_br - x_bl)
        ty = (y - y_bl) / (y_tl - y_bl)

        # Apply easing function for smoother transitions
        smoothtx = self.easeCurve(tx)
        smoothty = self.easeCurve(ty)

        # Get gradient vectors at grid corners
        v_bl = self.vectors[:, i, j]
        v_br = self.vectors[:, i + 1, j]
        v_tl = self.vectors[:, i, j + 1]
        v_tr = self.vectors[:, i + 1, j + 1]

        # Compute dot products between gradient and distance vectors
        s_bl = np.dot(v_bl, [x - x_bl, y - y_bl])
        s_br = np.dot(v_br, [x - x_br, y - y_br])
        s_tl = np.dot(v_tl, [x - x_tl, y - y_tl])
        s_tr = np.dot(v_tr, [x - x_tr, y - y_tr])

        # Interpolate between dot products
        d_lower = lerp(s_bl, s_br, smoothtx)
        d_upper = lerp(s_tl, s_tr, smoothtx)

        # Final interpolation
        noise_value = lerp(d_lower, d_upper, smoothty)
        # Normalize to 0-1 range
        return (noise_value + 1) / 2


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

        # Dont show the lines
        for line in self.lines:
            line.visible = False
        # Keep track of the next line
        self.next_line_index = 0

    def add_vertex(self, vertex: np.ndarray) -> None:  # 5 points
        self.vertices = np.vstack((self.vertices, np.array([vertex])))

    def delete_vertex(self, index: int) -> None:  # 5 points
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
        # ENDRE FOR NORMALS OGSÅ?
        self.triangle_indices = np.vstack((self.triangle_indices, np.array([triangle])))
        self.emit_array = np.vstack((self.emit_array, np.array([emit])))
        self.reflect_array = np.vstack((self.reflect_array, np.array([reflect])))

    def delete_triangle(self, index: int) -> None:  # 5 points
        # Previously named: delete_element

        # ENDRE FOR NORMALS OGSÅ?
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
        center = np.array([1, 1, 0])
        projected += center
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

    def build_triangles(
        self,
        window_width: int,
        window_height: int,
    ) -> None:
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
            # Calculate how many powers of 2 would be needed to represent this size
            # Round up to ensure we have enough space
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

    def update_form_factors(
        self, raytrace: bool = False, n_rays: int = 10
    ) -> None:  # 5 points
        if raytrace:
            # Use ray tracing for form factor calculation
            rng = np.random.default_rng()  # Create a random number generator
            # n_rays = 100  # Number of rays to cast per triangle
            self.form_factor_array = form_factors_raytrace(
                self.vertices, self.triangle_indices, n_rays, rng
            )
        else:
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
    cube.update_form_factors()
    cube.colors = cube.find_colors(num_iterations=25)

    noise_class = PerlinNoise(width=g_window_width, height=g_window_height)

    noise_values = noise_class.noise2(x=1, y=2)
    transparency = 50 * np.ones((g_window_width, g_window_height))

    # this makes the noise greyscale (sets RGB to be the same) and adds the alpha channel
    noise_values = np.dstack(
        [noise_values, noise_values, noise_values, transparency]
    ).astype(np.uint32)
    noise_values = noise_values.reshape(g_window_width * g_window_height * 4)

    # noise_data = (GLubyte * noise_values.size)(*noise_values.astype("uint8"))
    # noise = pyglet.image.ImageData(
    #     g_window_width, g_window_height, "RGBA", noise_data, pitch=g_window_width * 3
    # )

    # sprite = pyglet.sprite.Sprite(img=noise)

    # Initial preparation in case the drawing function is called
    cube.prepare()

    # Draw to the screen
    @g_window.event
    def on_draw() -> None:
        g_window.clear()
        # sprite.draw()
        cube.batch.draw()

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

        # Prepare for a new transformation
        cube.prepare()

        # Rotate all the quaternions
        cube.rotate(qx * qy * qz)

        # Translate the relative postition
        cube.translate(cube.position)

        # Building the needed triangles to the screen
        cube.build_triangles(window_width=g_window_width, window_height=g_window_height)

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
