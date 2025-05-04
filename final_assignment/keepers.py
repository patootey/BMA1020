
    def build_triangles(
        self,
        window_width: int,
        window_height: int,
        # NEW
        # vis_check=True,
    ) -> None:
        """
        Builds all the triangles for the viewport.
        The triangles will be rendered by pyglet.

        :param window_width: Window width.
        :param window_height: Window heigth.
        """
     
        # Originally named: draw
        # Draw the model to the screen
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
        # camera_position = np.array([0, 0, 0])
        # Rebuild the triangles for each edge
        for triangle_index in sorted_indices:

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
