import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)


def midpoint_2d(heightmap: np.ndarray, i: int, j: int, grid_n: int, rand_scale: float):
    # Calculate the index of the last element in the square
    last_index = 2**grid_n

    # Get original corner values
    top_left = heightmap[i, j]
    top_right = heightmap[i, j + last_index]
    bottom_left = heightmap[i + last_index, j]
    bottom_right = heightmap[i + last_index, j + last_index]

    if last_index + 1 == heightmap.shape[0]:
        # Fill with nan values, because to be able to determine if a value is already set
        heightmap[:, :] = np.nan
        # Rewrite corners
        heightmap[i, j] = top_left
        heightmap[i, j + last_index] = top_right
        heightmap[i + last_index, j] = bottom_left
        heightmap[i + last_index, j + last_index] = bottom_right

    if grid_n < 1:
        return

    half = last_index // 2

    # Calculate midpoint of the square
    center_i = i + half
    center_j = j + half

    corners = np.array([top_left, top_right, bottom_left, bottom_right])
    # Calculate average
    mid = np.mean(corners) + rand_scale * (rng.random() - 0.5)
    heightmap[center_i, center_j] = mid

    # Only fill in sides if they are empty
    # Left side
    if np.isnan(heightmap[center_i, j]):
        heightmap[center_i, j] = np.mean([top_left, top_right, mid]) + rand_scale * (
            rng.random() - 0.5
        )
    # Bottom side
    if np.isnan(heightmap[i + last_index, center_j]):
        heightmap[i + last_index, center_j] = np.mean(
            [top_right, bottom_right, mid]
        ) + rand_scale * (rng.random() - 0.5)
    # Right side
    if np.isnan(heightmap[center_i, j + last_index]):
        heightmap[center_i, j + last_index] = np.mean(
            [bottom_left, bottom_right, mid]
        ) + rand_scale * (rng.random() - 0.5)
    # Top side
    if np.isnan(heightmap[i, center_j]):
        heightmap[i, center_j] = np.mean([top_left, bottom_left, mid]) + rand_scale * (
            rng.random() - 0.5
        )

    # Recursively apply to the four quadrants
    if grid_n > 1:
        # Decrease scale for lower levels
        rand_scale_max = max(rand_scale * 0.6, 0.1)
        midpoint_2d(heightmap, i, j, (grid_n - 1), rand_scale_max)  # Top-left
        midpoint_2d(heightmap, i, j + half, (grid_n - 1), rand_scale_max)  # Top-right
        midpoint_2d(heightmap, i + half, j, (grid_n - 1), rand_scale_max)  # Bottom-left
        midpoint_2d(
            heightmap, i + half, j + half, (grid_n - 1), rand_scale_max
        )  # Bottom-right
    return


# Test code that shows that the algorithm works
n = 6  # Creates a (2^6 + 1) x (2^6 + 1) = 65x65 grid
size = 2**n + 1

# Initialize array with zeros
terrain = np.zeros((size, size))

# Set the corner values (you can experiment with different values)
terrain[0, 0] = 0.1
terrain[0, size - 1] = 0.2
terrain[size - 1, 0] = 0.8
terrain[size - 1, size - 1] = 0.3

# Apply the midpoint displacement algorithm
midpoint_2d(terrain, 0, 0, n, rand_scale=1.0)

# Create a simple 2D plot with matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(terrain, cmap="terrain")
plt.colorbar(label="Elevation")
plt.title("Terrain Generated with Diamond-Square Algorithm")
plt.show()
