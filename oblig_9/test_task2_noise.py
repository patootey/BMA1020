import numpy as np
import matplotlib.pyplot as plt
from task_2 import grid_x_size, grid_y_size, height, noise, width, vectors


def generate_noise_image(width, height, x_grid, y_grid, vectors, scale=1.0):
    """Generate a 2D image of Perlin noise"""
    img = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            img[y, x] = noise(x * scale, y * scale, x_grid, y_grid, vectors)

    # Normalize to [0, 1] for display
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


noise_img = generate_noise_image(
    width, height, grid_x_size, grid_y_size, vectors, scale=1.0
)
plt.figure(figsize=(10, 8))
plt.imshow(noise_img, cmap="gray")
plt.title("Perlin Noise")
plt.colorbar()
plt.show()
