import numpy as np
import matplotlib.pyplot as plt

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
# Remember rng, the random number generator from the first question?
# Use rng to generate an (m x n) array of 2d unit vectors, i.e. a (2 x m x n) array
# such that np.sum(vectors ** 2, axis=0) is an array of ones

# Use rng to generate an (m x n) array of 2d unit vectors

# Got from Caroline
angles = 2 * np.pi * rng.random((grid_x_size, grid_y_size))
vectors = np.array([np.cos(angles), np.sin(angles)])


# Found on web
def easeCurve(t):  # to smooth the noise (f(t) from PDF)
    return 3 * t**2 - 2 * t**3


def lerp(a, b, t):
    return a * (1 - t) + b * t


def noise(x, y, x_grid, y_grid, vectors: np.ndarray):
    # Define i and j
    i = np.clip(int(x / cell_width), 0, grid_x_size - 2)
    j = np.clip(int(y / cell_height), 0, grid_y_size - 2)

    # tl = top left, tr = top right, bl = bottom left, br = bottom right
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
    smoothtx = easeCurve(tx)
    smoothty = easeCurve(ty)

    # Find vectors to corners
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
