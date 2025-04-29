import numpy as np

# import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)


def midpoint_2d(A, i, j, n, rand_scale) -> None:

    size = 2**n

    if n < 1:
        return A[i : i + 1, j : j + 1]

    half = size // 2

    # DIAMOND STEP
    # Calculate midpoint of the square
    mid_i = i + half
    mid_j = j + half

    # Set midpoint as average
    corners = [
        A[i, j],
        A[i, j + size],
        A[i + size, j],
        A[i + size, j + size],
    ]
    A[mid_i, mid_j] = (np.sum(corners)) / 4 + rand_scale * size * (rng.random() - 0.5)

    A[i, mid_j] = (corners[0] + corners[1]) / 2.0 + rand_scale * size * (
        rng.random() - 0.5
    )
    A[i + size, mid_j] = (corners[2] + corners[3]) / 2.0 + rand_scale * size * (
        rng.random() - 0.5
    )
    A[mid_i, j] = (corners[0] + corners[2]) / 2.0 + rand_scale * size * (
        rng.random() - 0.5
    )
    A[mid_i, j + size] = (corners[1] + corners[3]) / 2.0 + rand_scale * size * (
        rng.random() - 0.5
    )

    # Recursively apply to the four squares
    if n > 1:
        midpoint_2d(A, i, j, n - 1, rand_scale)  # Top-left
        midpoint_2d(A, i, j + half, n - 1, rand_scale)  # Top-right
        midpoint_2d(A, i + half, j, n - 1, rand_scale)  # Bottom-left
        midpoint_2d(A, i + half, j + half, n - 1, rand_scale)  # Bottom-right

    return A[i : i + size + 1, j : j + size + 1]


# # Example usage
# n = 6  # Creates a (2^6 + 1) x (2^6 + 1) = 65x65 grid
# size = 2**n + 1

# # Initialize array with zeros
# terrain = np.zeros((size, size))

# # Set the corner values (you can experiment with different values)
# terrain[0, 0] = 0
# terrain[0, size - 1] = 0
# terrain[size - 1, 0] = 0
# terrain[size - 1, size - 1] = 0

# # Apply the midpoint displacement algorithm
# midpoint_2d(terrain, 0, 0, n, rand_scale=1.0)

# # Create a simple 2D plot with matplotlib
# plt.figure(figsize=(10, 8))
# plt.imshow(terrain, cmap="terrain")
# plt.colorbar(label="Elevation")
# plt.title("Terrain Generated with Diamond-Square Algorithm")
# plt.show()


# # Create a small test matrix (5x5 for n=2)
# n = 2  # 2^2 + 1 = 5
# size = 2**n + 1
# A = np.zeros((size, size))

# # Set corner values
# A[0, 0] = 1.0  # Top-left
# A[0, size - 1] = 2.0  # Top-right
# A[size - 1, 0] = 3.0  # Bottom-left
# A[size - 1, size - 1] = 4.0  # Bottom-right

# print("Initial matrix:")
# print(A)
# print("\nAfter midpoint displacement:")
# result = midpoint_2d(A, 0, 0, n, 0.5)
# print(result)

# # Verify size
# print(f"\nMatrix shape: {result.shape}")
# # Verify corners remained unchanged
# print("\nCorner values:")
# print(f"Top-left: {result[0,0]}")
# print(f"Top-right: {result[0,-1]}")
# print(f"Bottom-left: {result[-1,0]}")
# print(f"Bottom-right: {result[-1,-1]}")
