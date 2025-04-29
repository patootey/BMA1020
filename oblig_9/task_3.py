# Generate n unit vectors with 'Lambertian' distribution
# from a triangular element

# and n points in the triangle
import numpy as np
import matplotlib.pyplot as plt


def random_rays(rng, x0, x1, x2, n):
    # rng generator; x0,x1,x2 vertices of triangle; n number of rays
    # first find the normal vectors
    u = x1 - x0
    v = x2 - x0
    normal = np.cross(u, v)
    length = np.linalg.norm(normal)
    normalize = normal / length

    # now an orthonormal basis for surface, with third component the normal component
    u1 = u / np.linalg.norm(u)
    v1 = np.cross(normalize, u1)

    """
    A couple of corrections in the code random_rays(rng, x0, x1, x2, n) provided in 9/3: ... normal = N / n # not N = N / n, this you seem to have noticed ... 
    B = np.array([u1,v1,normal]).T # the transpose is missing, which is likely harder to spot I will give full marks to any submissions missing the transpose. 
    But you should correct it when you come to integrate the code in the final programming assignment, otherwise things won't look as good.
    """
    # noe charles sa
    B = np.array([u1, v1, normal]).T
    # B = np.column_stack((u1, v1, normalize))

    # generate n random polar and azimuthal angles
    # https://en.wikipedia.org/wiki/Lambertian_reflectance
    theta = np.pi * rng.random(size=n) / 2
    phi = 2 * np.pi * rng.random(size=n)

    rays_local = np.array(
        [np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)]
    )
    # transform to 3d coordinates in the standard basis
    r = B @ rays_local
    # Now generate n points inside the triangle
    # this bit you can do yourself!

    r1 = rng.random(size=n)
    r2 = rng.random(size=n)

    # Uniformly distributed random points inside a triangle.

    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    x = (
        (1 - r1 - r2)[:, np.newaxis] * x0
        + r1[:, np.newaxis] * x1
        + r2[:, np.newaxis] * x2
    )

    return x, r.T


def ray_triangle_intersection(ray_origin, ray_direction, vertex0, vertex1, vertex2):
    # USIKKER OM DETTE ER RIKIG, fant på nett
    # Kanskje ha med epsilon? usikker på hva det engeltilig her

    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    edge_a = vertex1 - vertex0
    edge_b = vertex2 - vertex0

    # Calculate determinant
    ray_cross_edge_b = np.cross(ray_direction, edge_b)
    determinant = np.dot(edge_a, ray_cross_edge_b)

    if determinant == 0.0:
        return None  # Ray is parallel to the triangle plane

    inv_det = 1.0 / determinant

    # Calculate distance from vertex0 to ray origin
    origin_to_vertex = ray_origin - vertex0

    u = inv_det * np.dot(origin_to_vertex, ray_cross_edge_b)
    if u < 0.0 or u > 1.0:
        return None

    origin_cross_edge_a = np.cross(origin_to_vertex, edge_a)
    v = inv_det * np.dot(ray_direction, origin_cross_edge_a)

    if v < 0.0 or (u + v) > 1.0:
        return None

    t = inv_det * np.dot(edge_b, origin_cross_edge_a)

    if t < 0.0:
        return None  # Intersection is behind the ray start

    # Compute the exact intersection point
    intersection_point = ray_origin + t * ray_direction
    # kan vel egt bare heller retunere True og false? Skal jeg bruke punktet til noe?
    return intersection_point


def form_factors_raytrace(vertices, ray_indices, n_rays, rng):
    ray_size = np.size(ray_indices, axis=0)
    form_factor = np.zeros((ray_size, ray_size))
    for i, ray_index_i in enumerate(ray_indices):
        v0_i = vertices[ray_index_i[0]]
        v1_i = vertices[ray_index_i[1]]
        v2_i = vertices[ray_index_i[2]]
        # generate n rays from random points x in element i, in random directions r
        x, r = random_rays(rng, v0_i, v1_i, v2_i, n_rays)

        # find out which element (if any) each ray hits
        # add 1/n to F[i,j] for each ray from i hitting j
        for ray_idx in range(n_rays):
            ray_origin = x[ray_idx]
            ray_direction = r[ray_idx]

            # found = False
            for j, ray_index_j in enumerate(ray_indices):
                if i == j:  # Skip self-intersection
                    continue

                # Get vertices of element j
                v0_j = vertices[ray_index_j[0]]
                v1_j = vertices[ray_index_j[1]]
                v2_j = vertices[ray_index_j[2]]

                intersection = ray_triangle_intersection(
                    ray_origin, ray_direction, v0_j, v1_j, v2_j
                )

                # usikker hva jeg skal sende in?
                if intersection is not None:
                    form_factor[i, j] += 1.0 / n_rays
                    # found = True
                    break

    return form_factor


# rng = np.random.default_rng(12345)


def test_ray_generation():
    """Test and visualize the random ray generation from a triangle."""
    # Set random seed for reproducibility
    rng = np.random.default_rng(12345)

    # Create a triangle in 3D space (but we'll visualize in 2D)
    x0 = np.array([0, 0, 0])
    x1 = np.array([1, 0, 0])
    x2 = np.array([0, 1, 0])

    # Generate random points and ray directions
    n_rays = 50
    points, directions = random_rays(rng, x0, x1, x2, n_rays)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the triangle and points - XY plane view
    ax1.plot([x0[0], x1[0], x2[0], x0[0]], [x0[1], x1[1], x2[1], x0[1]], "k-")
    ax1.fill([x0[0], x1[0], x2[0]], [x0[1], x1[1], x2[1]], alpha=0.2)
    ax1.scatter(points[:, 0], points[:, 1], color="blue", s=10)

    # Plot the first few ray directions as arrows in XY plane
    for i in range(min(15, n_rays)):
        ax1.arrow(
            points[i, 0],
            points[i, 1],
            directions[i, 0] * 0.2,
            directions[i, 1] * 0.2,
            head_width=0.02,
            head_length=0.03,
            fc="red",
            ec="red",
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Top View (XY Plane)")
    ax1.set_aspect("equal")
    ax1.grid(True)

    # Plot the Z component of the rays - side view
    ax2.plot([x0[0], x1[0], x2[0], x0[0]], [x0[2], x1[2], x2[2], x0[2]], "k-")
    ax2.scatter(points[:, 0], points[:, 2], color="blue", s=10)

    # Plot the first few ray directions as arrows in XZ plane
    for i in range(min(15, n_rays)):
        ax2.arrow(
            points[i, 0],
            points[i, 2],
            directions[i, 0] * 0.2,
            directions[i, 2] * 0.2,
            head_width=0.02,
            head_length=0.03,
            fc="red",
            ec="red",
        )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Side View (XZ Plane)")
    ax2.set_aspect("equal")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("ray_generation.png")

    print(f"Generated {n_rays} rays from random points in the triangle")
    print(
        f"Average direction vector length: {np.mean(np.linalg.norm(directions, axis=1)):.4f}"
    )
    return fig


def test_ray_triangle_intersection():
    """Test and visualize ray-triangle intersection in 2D views."""
    # Create a simple scene with two triangles
    # Triangle 1 in xy-plane
    t1_v0 = np.array([0, 0, 0])
    t1_v1 = np.array([1, 0, 0])
    t1_v2 = np.array([0, 1, 0])

    # Triangle 2 offset and tilted
    t2_v0 = np.array([0.25, 0.25, 1])
    t2_v1 = np.array([0.75, 0.25, 1])
    t2_v2 = np.array([0.5, 0.75, 0.8])

    # Define a ray that should intersect triangle 2
    ray_origin = np.array([0.4, 0.4, 0])
    ray_direction = np.array([0.1, 0.1, 1])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize

    # Calculate intersection
    intersection = ray_triangle_intersection(
        ray_origin, ray_direction, t2_v0, t2_v1, t2_v2
    )

    # Create a figure with three subplots for different views
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Top view (XY plane)
    ax1.plot(
        [t1_v0[0], t1_v1[0], t1_v2[0], t1_v0[0]],
        [t1_v0[1], t1_v1[1], t1_v2[1], t1_v0[1]],
        "b-",
        label="Triangle 1",
    )
    ax1.fill(
        [t1_v0[0], t1_v1[0], t1_v2[0]], [t1_v0[1], t1_v1[1], t1_v2[1]], "b", alpha=0.2
    )

    ax1.plot(
        [t2_v0[0], t2_v1[0], t2_v2[0], t2_v0[0]],
        [t2_v0[1], t2_v1[1], t2_v2[1], t2_v0[1]],
        "g-",
        label="Triangle 2",
    )
    ax1.fill(
        [t2_v0[0], t2_v1[0], t2_v2[0]], [t2_v0[1], t2_v1[1], t2_v2[1]], "g", alpha=0.2
    )

    ax1.plot(ray_origin[0], ray_origin[1], "ro", ms=6, label="Ray Origin")

    if intersection is not None:
        ax1.plot(
            [ray_origin[0], intersection[0]],
            [ray_origin[1], intersection[1]],
            "r-",
            label="Ray Path",
        )
        ax1.plot(intersection[0], intersection[1], "mo", ms=6, label="Intersection")
    else:
        # Show ray direction for some distance if no intersection
        end_point = ray_origin + ray_direction * 1.5
        ax1.plot(
            [ray_origin[0], end_point[0]],
            [ray_origin[1], end_point[1]],
            "r-",
            label="Ray Path",
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Top View (XY Plane)")
    ax1.grid(True)
    ax1.set_aspect("equal")

    # Front view (XZ plane)
    ax2.plot(
        [t1_v0[0], t1_v1[0], t1_v2[0], t1_v0[0]],
        [t1_v0[2], t1_v1[2], t1_v2[2], t1_v0[2]],
        "b-",
    )
    ax2.fill(
        [t1_v0[0], t1_v1[0], t1_v2[0]], [t1_v0[2], t1_v1[2], t1_v2[2]], "b", alpha=0.2
    )

    ax2.plot(
        [t2_v0[0], t2_v1[0], t2_v2[0], t2_v0[0]],
        [t2_v0[2], t2_v1[2], t2_v2[2], t2_v0[2]],
        "g-",
    )
    ax2.fill(
        [t2_v0[0], t2_v1[0], t2_v2[0]], [t2_v0[2], t2_v1[2], t2_v2[2]], "g", alpha=0.2
    )

    ax2.plot(ray_origin[0], ray_origin[2], "ro", ms=6)

    if intersection is not None:
        ax2.plot(
            [ray_origin[0], intersection[0]], [ray_origin[2], intersection[2]], "r-"
        )
        ax2.plot(intersection[0], intersection[2], "mo", ms=6)
    else:
        end_point = ray_origin + ray_direction * 1.5
        ax2.plot([ray_origin[0], end_point[0]], [ray_origin[2], end_point[2]], "r-")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Front View (XZ Plane)")
    ax2.grid(True)
    ax2.set_aspect("equal")

    # Side view (YZ plane)
    ax3.plot(
        [t1_v0[1], t1_v1[1], t1_v2[1], t1_v0[1]],
        [t1_v0[2], t1_v1[2], t1_v2[2], t1_v0[2]],
        "b-",
    )
    ax3.fill(
        [t1_v0[1], t1_v1[1], t1_v2[1]], [t1_v0[2], t1_v1[2], t1_v2[2]], "b", alpha=0.2
    )

    ax3.plot(
        [t2_v0[1], t2_v1[1], t2_v2[1], t2_v0[1]],
        [t2_v0[2], t2_v1[2], t2_v2[2], t2_v0[2]],
        "g-",
    )
    ax3.fill(
        [t2_v0[1], t2_v1[1], t2_v2[1]], [t2_v0[2], t2_v1[2], t2_v2[2]], "g", alpha=0.2
    )

    ax3.plot(ray_origin[1], ray_origin[2], "ro", ms=6)

    if intersection is not None:
        ax3.plot(
            [ray_origin[1], intersection[1]], [ray_origin[2], intersection[2]], "r-"
        )
        ax3.plot(intersection[1], intersection[2], "mo", ms=6)
    else:
        end_point = ray_origin + ray_direction * 1.5
        ax3.plot([ray_origin[1], end_point[1]], [ray_origin[2], end_point[2]], "r-")

    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_title("Side View (YZ Plane)")
    ax3.grid(True)
    ax3.set_aspect("equal")

    # Add legend to the first subplot only
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("ray_intersection.png")

    if intersection is not None:
        print(f"Ray intersection found at: {intersection}")
    else:
        print("No intersection found")

    return fig


def test_form_factors():
    """Test the form factor calculation between two triangles."""
    # Create a scene with two parallel triangles facing each other
    vertices = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 1],  # 3
            [1, 0, 1],  # 4
            [0, 1, 1],  # 5
        ]
    )

    # Triangle indices
    ray_indices = np.array(
        [
            [0, 1, 2],  # Triangle 1 (bottom)
            [3, 4, 5],  # Triangle 2 (top)
        ]
    )

    # Calculate form factors
    rng = np.random.default_rng(12345)
    n_rays = 1000
    form_factors = form_factors_raytrace(vertices, ray_indices, n_rays, rng)

    # Print results
    print("Form Factor Matrix:")
    print(form_factors)

    # Plot the triangles and some sample rays
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot triangles - XY plane view with Z offset
    tri1 = vertices[ray_indices[0]]
    tri2 = vertices[ray_indices[1]]

    # First subplot - Side view (XZ plane)
    ax1.plot(
        [tri1[0, 0], tri1[1, 0], tri1[2, 0], tri1[0, 0]],
        [tri1[0, 2], tri1[1, 2], tri1[2, 2], tri1[0, 2]],
        "b-",
        label="Triangle 1",
    )
    ax1.fill(
        [tri1[0, 0], tri1[1, 0], tri1[2, 0]],
        [tri1[0, 2], tri1[1, 2], tri1[2, 2]],
        "b",
        alpha=0.2,
    )

    ax1.plot(
        [tri2[0, 0], tri2[1, 0], tri2[2, 0], tri2[0, 0]],
        [tri2[0, 2], tri2[1, 2], tri2[2, 2], tri2[0, 2]],
        "g-",
        label="Triangle 2",
    )
    ax1.fill(
        [tri2[0, 0], tri2[1, 0], tri2[2, 0]],
        [tri2[0, 2], tri2[1, 2], tri2[2, 2]],
        "g",
        alpha=0.2,
    )

    # Draw some sample rays
    sample_rays = 10
    for i in range(sample_rays):
        # Generate random point on triangle 1
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        point = (1 - r1 - r2) * tri1[0] + r1 * tri1[1] + r2 * tri1[2]

        # Generate random point on triangle 2
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        target = (1 - r1 - r2) * tri2[0] + r1 * tri2[1] + r2 * tri2[2]

        # Draw ray from point to target
        ax1.plot([point[0], target[0]], [point[2], target[2]], "r-", alpha=0.3)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_title("Side View (XZ Plane)")
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect("equal")

    # Second subplot - Top view (XY plane)
    ax2.plot(
        [tri1[0, 0], tri1[1, 0], tri1[2, 0], tri1[0, 0]],
        [tri1[0, 1], tri1[1, 1], tri1[2, 1], tri1[0, 1]],
        "b-",
        label="Triangle 1",
    )
    ax2.fill(
        [tri1[0, 0], tri1[1, 0], tri1[2, 0]],
        [tri1[0, 1], tri1[1, 1], tri1[2, 1]],
        "b",
        alpha=0.2,
    )

    ax2.plot(
        [tri2[0, 0], tri2[1, 0], tri2[2, 0], tri2[0, 0]],
        [tri2[0, 1], tri2[1, 1], tri2[2, 1], tri2[0, 1]],
        "g-",
        label="Triangle 2",
    )
    ax2.fill(
        [tri2[0, 0], tri2[1, 0], tri2[2, 0]],
        [tri2[0, 1], tri2[1, 1], tri2[2, 1]],
        "g",
        alpha=0.2,
    )

    # Draw the same rays in top view
    for i in range(sample_rays):
        # Generate random point on triangle 1
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        point = (1 - r1 - r2) * tri1[0] + r1 * tri1[1] + r2 * tri1[2]

        # Generate random point on triangle 2
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        target = (1 - r1 - r2) * tri2[0] + r1 * tri2[1] + r2 * tri2[2]

        # Draw ray from point to target
        ax2.plot([point[0], target[0]], [point[1], target[1]], "r-", alpha=0.3)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top View (XY Plane)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("form_factors.png")

    return fig


def test_complex_scene():
    """Test with a more complex scene with multiple triangles."""
    # Create a more complex scene with multiple triangles
    vertices = np.array(
        [
            # Floor
            [0, 0, 0],  # 0
            [2, 0, 0],  # 1
            [2, 2, 0],  # 2
            [0, 2, 0],  # 3
            # Wall 1 (left)
            [0, 0, 0],  # 4
            [0, 2, 0],  # 5
            [0, 2, 1],  # 6
            [0, 0, 1],  # 7
            # Wall 2 (back)
            [0, 0, 0],  # 8
            [2, 0, 0],  # 9
            [2, 0, 1],  # 10
            [0, 0, 1],  # 11
            # Ceiling light
            [0.5, 0.5, 1],  # 12
            [1.5, 0.5, 1],  # 13
            [1.5, 1.5, 1],  # 14
            [0.5, 1.5, 1],  # 15
        ]
    )

    # Triangle indices - splitting quads into triangles
    ray_indices = np.array(
        [
            # Floor (2 triangles)
            [0, 1, 2],
            [0, 2, 3],
            # Left wall (2 triangles)
            [4, 5, 6],
            [4, 6, 7],
            # Back wall (2 triangles)
            [8, 9, 10],
            [8, 10, 11],
            # Ceiling light (2 triangles)
            [12, 13, 14],
            [12, 14, 15],
        ]
    )

    # Calculate form factors
    rng = np.random.default_rng(12345)
    n_rays = 500
    form_factors = form_factors_raytrace(vertices, ray_indices, n_rays, rng)

    # Print results
    print("Form Factor Matrix for complex scene:")
    print(form_factors)

    # Plot the scene - display the room from a side view
    fig, ax = plt.subplots(figsize=(10, 8))

    # Function to plot triangles with specified colors
    def plot_triangle(ax, triangle, color="b", alpha=0.2, label=None):
        ax.plot(
            [triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 2], triangle[1, 2], triangle[2, 2], triangle[0, 2]],
            color + "-",
            label=label,
        )
        ax.fill(
            [triangle[0, 0], triangle[1, 0], triangle[2, 0]],
            [triangle[0, 2], triangle[1, 2], triangle[2, 2]],
            color,
            alpha=alpha,
        )

    # Plot triangles - side view (XZ plane, Y=1)
    # Floor
    plot_triangle(ax, vertices[ray_indices[0]], "b", 0.2, "Floor")
    plot_triangle(ax, vertices[ray_indices[1]], "b", 0.2)

    # Left wall
    plot_triangle(ax, vertices[ray_indices[2]], "g", 0.2, "Left Wall")
    plot_triangle(ax, vertices[ray_indices[3]], "g", 0.2)

    # Back wall
    plot_triangle(ax, vertices[ray_indices[4]], "r", 0.2, "Back Wall")
    plot_triangle(ax, vertices[ray_indices[5]], "r", 0.2)

    # Ceiling light
    plot_triangle(ax, vertices[ray_indices[6]], "y", 0.4, "Ceiling Light")
    plot_triangle(ax, vertices[ray_indices[7]], "y", 0.4)

    # Show some random rays from light to other surfaces
    light_indices = [6, 7]  # Indices of light triangles
    surface_indices = [0, 1, 2, 3, 4, 5]  # Indices of other surfaces

    # Generate 15 random rays from light to surfaces
    sample_rays = 15
    for i in range(sample_rays):
        # Choose a random light triangle
        light_idx = light_indices[rng.integers(0, len(light_indices))]
        light_tri = vertices[ray_indices[light_idx]]

        # Generate random point on light
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        light_point = (
            (1 - r1 - r2) * light_tri[0] + r1 * light_tri[1] + r2 * light_tri[2]
        )

        # Choose a random surface triangle
        surface_idx = surface_indices[rng.integers(0, len(surface_indices))]
        surface_tri = vertices[ray_indices[surface_idx]]

        # Generate random point on surface
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        surface_point = (
            (1 - r1 - r2) * surface_tri[0] + r1 * surface_tri[1] + r2 * surface_tri[2]
        )

        # Draw ray from light to surface
        ax.plot(
            [light_point[0], surface_point[0]],
            [light_point[2], surface_point[2]],
            "m-",
            alpha=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Side View of Room with Light (XZ Plane)")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("complex_scene.png")

    # Plot the form factor matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(form_factors, cmap="viridis")
    plt.colorbar(label="Form Factor")
    plt.title("Form Factor Matrix Heatmap")
    plt.xlabel("Receiving Triangle Index")
    plt.ylabel("Emitting Triangle Index")

    # Add text annotations in the heatmap
    for i in range(form_factors.shape[0]):
        for j in range(form_factors.shape[1]):
            plt.text(
                j,
                i,
                f"{form_factors[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if form_factors[i, j] < 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig("form_factor_heatmap.png")

    return fig


if __name__ == "__main__":
    print("Testing ray generation...")
    test_ray_generation()

    print("\nTesting ray-triangle intersection...")
    test_ray_triangle_intersection()

    print("\nTesting form factors between two triangles...")
    test_form_factors()

    print("\nTesting complex scene with multiple triangles...")
    test_complex_scene()

    print("\nAll tests completed!")


# ------------------------------------------------------------------------


# TROR DETTE ER FEIL, KAN VI SETTE OPP EGEN TEST PAPPA?

# # Set random seed for reproducibility
# rng = np.random.default_rng(42)

# # Create a simple scene with two parallel squares facing each other
# # Square 1: on the xy-plane at z=0
# # Square 2: on the xy-plane at z=1, facing down
# nodes = np.array(
#     [
#         [0, 0, 0],  # 0: bottom-left corner of square 1
#         [1, 0, 0],  # 1: bottom-right corner of square 1
#         [1, 1, 0],  # 2: top-right corner of square 1
#         [0, 1, 0],  # 3: top-left corner of square 1
#         [0, 0, 1],  # 4: bottom-left corner of square 2
#         [1, 0, 1],  # 5: bottom-right corner of square 2
#         [1, 1, 1],  # 6: top-right corner of square 2
#         [0, 1, 1],  # 7: top-left corner of square 2
#     ]
# )

# # Split each square into two triangles
# elements = np.array(
#     [
#         [0, 1, 2],  # Triangle 1 of square 1
#         [0, 2, 3],  # Triangle 2 of square 1
#         [4, 5, 6],  # Triangle 1 of square 2
#         [4, 6, 7],  # Triangle 2 of square 2
#     ]
# )

# # Number of rays to shoot
# n_rays = 10000

# # Calculate form factors
# F = form_factors_raytrace(nodes, elements, n_rays, rng)

# print("Form Factor Matrix:")
# print(F)

# # Verify row sums (should be <= 1)
# row_sums = np.sum(F, axis=1)
# print("\nRow sums (total form factors from each element):")
# print(row_sums)

# # Verify reciprocity relation: A_i * F_ij = A_j * F_ji
# # For our simple case with equal-sized triangles, we expect: F_ij ≈ F_ji
# print("\nReciprocity check (should be approximately equal):")
# print(f"F[0,2] = {F[0,2]}, F[2,0] = {F[2,0]}")
# print(f"F[1,3] = {F[1,3]}, F[3,1] = {F[3,1]}")

# # Visualize the scene
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# # Plot triangles
# for i, triangle in enumerate(elements):
#     v0, v1, v2 = nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]
#     ax.plot(
#         [v0[0], v1[0], v2[0], v0[0]],
#         [v0[1], v1[1], v2[1], v0[1]],
#         [v0[2], v1[2], v2[2], v0[2]],
#         "o-",
#         label=f"Triangle {i}",
#     )

# # Plot some rays for visualization
# n_vis_rays = 50  # Number of rays to visualize
# for i in range(2):  # Only from the first two triangles
#     el = elements[i]
#     x, r = random_rays(rng, nodes[el[0]], nodes[el[1]], nodes[el[2]], n_vis_rays)

#     for j in range(n_vis_rays):
#         # Ray origin
#         origin = x[j]
#         # Ray direction (extend a bit for visualization)
#         direction = r[j]
#         end = origin + direction * 1.5

#         ax.plot(
#             [origin[0], end[0]],
#             [origin[1], end[1]],
#             [origin[2], end[2]],
#             "r-",
#             alpha=0.1,
#         )

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Ray Tracing Scene with Two Parallel Squares")
# ax.legend()

# plt.tight_layout()
# plt.show()
