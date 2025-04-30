import numpy as np
import matplotlib.pyplot as plt

# from task_3 import form_factors_raytrace, random_rays
from task_3 import form_factors_raytrace, random_rays, ray_triangle_intersection

# TROR DETTE ER FEIL, KAN VI SETTE OPP EGEN TEST PAPPA?

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Create a simple scene with two parallel squares facing each other
# Square 1: on the xy-plane at z=0
# Square 2: on the xy-plane at z=1, facing down
nodes = np.array(
    [
        [0, 0, 0],  # 0: bottom-left corner of square 1
        [1, 0, 0],  # 1: bottom-right corner of square 1
        [1, 1, 0],  # 2: top-right corner of square 1
        [0, 1, 0],  # 3: top-left corner of square 1
        [0, 0, 1],  # 4: bottom-left corner of square 2
        [1, 0, 1],  # 5: bottom-right corner of square 2
        [1, 1, 1],  # 6: top-right corner of square 2
        [0, 1, 1],  # 7: top-left corner of square 2
    ]
)

# Split each square into two triangles
elements = np.array(
    [
        [0, 1, 2],  # Triangle 1 of square 1
        [0, 2, 3],  # Triangle 2 of square 1
        [4, 5, 6],  # Triangle 1 of square 2
        [4, 6, 7],  # Triangle 2 of square 2
    ]
)

# Number of rays to shoot
n_rays = 10000

# Calculate form factors
F = form_factors_raytrace(nodes, elements, n_rays, rng)

print("Form Factor Matrix:")
print(F)

# Verify row sums (should be <= 1)
row_sums = np.sum(F, axis=1)
print("\nRow sums (total form factors from each element):")
print(row_sums)

# Verify reciprocity relation: A_i * F_ij = A_j * F_ji
# For our simple case with equal-sized triangles, we expect: F_ij ≈ F_ji
print("\nReciprocity check (should be approximately equal):")
print(f"F[0,2] = {F[0, 2]}, F[2,0] = {F[2, 0]}")
print(f"F[1,3] = {F[1, 3]}, F[3,1] = {F[3, 1]}")

# Visualize the scene
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot triangles
for i, triangle in enumerate(elements):
    v0, v1, v2 = nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]
    ax.plot(
        [v0[0], v1[0], v2[0], v0[0]],
        [v0[1], v1[1], v2[1], v0[1]],
        [v0[2], v1[2], v2[2], v0[2]],
        "o-",
        label=f"Triangle {i}",
    )

# Plot some rays for visualization
n_vis_rays = 50  # Number of rays to visualize
for i in range(2):  # Only from the first two triangles
    el = elements[i]
    x, r = random_rays(rng, nodes[el[0]], nodes[el[1]], nodes[el[2]], n_vis_rays)

    for j in range(n_vis_rays):
        # Ray origin
        origin = x[j]
        # Ray direction (extend a bit for visualization)
        direction = r[j]
        end = origin + direction * 1.5

        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            "r-",
            alpha=0.1,
        )

        ax.plot(
            origin[0],
            origin[1],
            origin[2],
            "r+",
        )

        for k in range(2):
            hit_el = elements[k + 2]
            intersect_point = ray_triangle_intersection(
                origin, direction, nodes[hit_el[0]], nodes[hit_el[1]], nodes[hit_el[2]]
            )
            if intersect_point is not None:
                ax.plot(
                    intersect_point[0],
                    intersect_point[1],
                    intersect_point[2],
                    "*",
                )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")  # type: ignore
ax.set_title("Ray Tracing Scene with Two Parallel Squares")
ax.legend()

plt.tight_layout()
plt.show()
