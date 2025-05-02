# from matplotlib import pyplot as plt
from math import ceil
import numpy as np

from numba import njit


@njit
def ease_curve(t):
    # Perlins ease curve
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit
def lerp(a, b, t):
    # Linear interpolation
    return a * (1 - t) + b * t


def noise(width: int, height: int, grid_size: tuple[int, int] | None = None):

    if grid_size is None:
        grid_x_size = ceil(width / 60) + 1
        grid_y_size = ceil(height / 60) + 1

        cell_width = width // (grid_x_size - 1)
        cell_height = height // (grid_y_size - 1)
    else:
        grid_x_size, grid_y_size = grid_size

        cell_width = width // (grid_x_size - 1)
        cell_height = height // (grid_y_size - 1)

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, width, grid_x_size), np.linspace(0, height, grid_y_size)
    )

    angles = (
        2
        * np.pi
        * np.random.rand(grid_x_size * grid_y_size).reshape((grid_x_size, grid_y_size))
    )
    vectors = np.array([np.cos(angles), np.sin(angles)])

    def noise_element(x: float, y: float):
        # Define i and j
        i = np.clip(int(x / cell_width), 0, grid_x_size - 2)
        j = np.clip(int(y / cell_height), 0, grid_y_size - 2)

        # tl = top left, tr = top right, bl = bottom left, br = bottom right
        # Shape of x_grid/y_grid is (y, x)
        x_bl = x_grid[j, i]
        y_bl = y_grid[j, i]
        x_br = x_grid[j, i + 1]
        y_br = y_grid[j, i + 1]
        x_tl = x_grid[j + 1, i]
        y_tl = y_grid[j + 1, i]
        x_tr = x_grid[j + 1, i + 1]
        y_tr = y_grid[j + 1, i + 1]

        tx = (x - x_bl) / (x_br - x_bl)
        ty = (y - y_bl) / (y_tl - y_bl)

        # Smooth the noise
        smoothtx = ease_curve(tx)
        smoothty = ease_curve(ty)

        # Find vectors to corners
        # Shape of vectors is (2, x, y)
        v_bl = vectors[:, i, j]
        v_br = vectors[:, i + 1, j]
        v_tl = vectors[:, i, j + 1]
        v_tr = vectors[:, i + 1, j + 1]

        # Compute dot products between corner vectors and distance vectors
        s_bl = np.dot(v_bl, [x - x_bl, y - y_bl])
        s_br = np.dot(v_br, [x - x_br, y - y_br])
        s_tl = np.dot(v_tl, [x - x_tl, y - y_tl])
        s_tr = np.dot(v_tr, [x - x_tr, y - y_tr])

        # Interpolate between dot products
        d_lower = lerp(s_bl, s_br, smoothtx)
        d_upper = lerp(s_tl, s_tr, smoothtx)

        noise = lerp(d_lower, d_upper, smoothty)
        return noise

    values = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            values[y, x] = noise_element(x, y)

    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    return values


# noise_img = noise(600, 480)
# plt.figure(figsize=(10, 8))
# plt.imshow(noise_img, cmap="gray")
# plt.title("Perlin Noise")
# plt.colorbar()
# plt.show()
