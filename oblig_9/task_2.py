import numpy as np
import matplotlib.pyplot as plt

# choose your own values:
width = 600
height = 480

rng = np.random.default_rng(12345)

# choose your own values:
m = 11
n = 9

# generates the grid by
x_grid, y_grid = np.meshgrid(np.linspace(0, width, m), np.linspace(0, height, n))

# Remember rng, the random number generator from the first question?
# Use rng to generate an (m x n) array of 2d unit vectors, i.e. a (2 x m x n) array
# such that np.sum(vectors ** 2, axis=0) is an array of ones

# Use rng to generate an (m x n) array of 2d unit vectors

# Got from Caroline
angles = 2 * np.pi * rng.random((m, n))
vectors = np.array([np.cos(angles), np.sin(angles)])


# Found on web
def easeCurve(t):  # to smooth the noise (f(t) from PDF)
    return 3 * t**2 - 2 * t**3


def noiseCaroline(x, y, x_grid, y_grid, vectors: np.ndarray):
    i = min(int(x * (m - 1) / width), m - 2)
    j = min(int(y * (n - 1) / height), n - 2)

    # get the four corner coordinates of the cell

    x0, y0 = x_grid[j, i], y_grid[j, i]  # lower left
    x1, y1 = x_grid[j, i + 1], y_grid[j, i + 1]  # lower right
    x2, y2 = x_grid[j + 1, i], y_grid[j + 1, i]  # upper left
    x3, y3 = x_grid[j + 1, i + 1], y_grid[j + 1, i + 1]  # upper right

    # calculate the relative position in the cell --> help from TA
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y2 - y0)

    # find vectors in the corners
    vec0 = vectors[:, i, j]
    vec1 = vectors[:, i + 1, j]
    vec2 = vectors[:, i, j + 1]
    vec3 = vectors[:, i + 1, j + 1]

    # find the corner noises
    s0 = np.dot(vec0, [x - x0, y - y0])
    s1 = np.dot(vec1, [x - x1, y - y1])
    s2 = np.dot(vec2, [x - x2, y - y2])
    s3 = np.dot(vec3, [x - x3, y - y3])

    # interpolate using the easeCurve function in x-direction
    txSmooth = easeCurve(tx)
    Edgebottom = s0 + txSmooth * (s1 - s0)  # interpolate bottom edge
    Edgetop = s2 + txSmooth * (s3 - s2)  # interpolate top edge

    # and y-direction
    tySmooth = easeCurve(ty)
    noise = Edgebottom + tySmooth * (Edgetop - Edgebottom)

    return noise


def noise(x, y, x_grid, y_grid, vectors: np.ndarray):

    cell_width = width / (m - 1)
    cell_height = height / (n - 1)

    # Find the grid indices
    i = int(x / cell_width)
    j = int(y / cell_height)

    # Fikset clipping
    i = np.clip(i, 0, m - 2)
    j = np.clip(j, 0, n - 2)

    # Virker som det er feil her
    tx = (x - x_grid[j, i]) / cell_width
    ty = (y - y_grid[j, i]) / cell_height

    # Hvorfor har caroline plustelig motsatt her?
    # x0, y0 = x_grid[j, i], y_grid[j, i]  # lower left
    # x1, y1 = x_grid[j, i + 1], y_grid[j, i + 1]  # lower right
    # x2, y2 = x_grid[j + 1, i], y_grid[j + 1, i]  # upper left
    # x3, y3 = x_grid[j + 1, i + 1], y_grid[j + 1, i + 1]  # upper right

    # # calculate the relative position in the cell --> help from TA
    # tx = (x - x0) / (x1 - x0)
    # ty = (y - y0) / (y2 - y0)

    tx = easeCurve(tx)
    ty = easeCurve(ty)

    v1 = vectors[:, i, j]
    v2 = vectors[:, i, j + 1]
    v3 = vectors[:, i + 1, j]
    v4 = vectors[:, i + 1, j + 1]
    # Compute the distance
    d1 = np.array([tx, ty])  # Vector to bottom-left
    d2 = np.array([tx - 1, ty])  # Vector to bottom-right
    d3 = np.array([tx, ty - 1])  # Vector to top-left
    d4 = np.array([tx - 1, ty - 1])  # Vector to top-right

    # Compute dot products between corner vectors and distance vectors
    s1 = np.dot(v1, d1)
    s2 = np.dot(v2, d2)
    s3 = np.dot(v3, d3)
    s4 = np.dot(v4, d4)

    # carolinme sin måte
    # s0 = np.dot(vec0, [x - x0, y - y0])
    # s1 = np.dot(vec1, [x - x1, y - y1])
    # s2 = np.dot(vec2, [x - x2, y - y2])
    # s3 = np.dot(vec3, [x - x3, y - y3])

    # Followed notes? wrong?
    d1 = s1 * tx + (1 - tx) * s2
    d2 = s3 * tx * (1 - tx) * s4

    noise_i_j = d1 * ty + (1 - ty) * d2
    return noise_i_j


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
