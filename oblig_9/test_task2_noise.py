import numpy as np
import matplotlib.pyplot as plt
from task_2 import noise

# choose your own values:
width = 600
height = 480

rng = np.random.default_rng(12345)
# choose your own values:
grid_x_size = 11
grid_y_size = 9

# generates the grid by
x_grid, y_grid = np.meshgrid(
    np.linspace(0, width, grid_x_size), np.linspace(0, height, grid_y_size)
)


cell_width = width // (grid_x_size - 1)
cell_height = height // (grid_y_size - 1)

# Use rng to generate an (m x n) array of 2d unit vectors, i.e. a (2 x m x n) array
angles = 2 * np.pi * rng.random((grid_x_size, grid_y_size))
vectors = np.array([np.cos(angles), np.sin(angles)])


def generate_noise_image(width, height, x_grid, y_grid, vectors, scale=1.0):
    """Generate a 2D image of Perlin noise"""
    img = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            img[y, x] = noise(x * scale, y * scale, x_grid, y_grid, vectors)

    # Normalize to [0, 1] for display
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


noise_img = generate_noise_image(width, height, x_grid, y_grid, vectors, scale=1.0)
plt.figure(figsize=(10, 8))
plt.imshow(noise_img, cmap="gray")
plt.title("Perlin Noise")
plt.colorbar()
plt.show()
