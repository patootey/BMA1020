import numpy as np

from task_1 import midpoint_2d

import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

# Example usage
n = 6  # Creates a (2^6 + 1) x (2^6 + 1) = 65x65 grid
size = 2**n + 1

# Initialize array with zeros
terrain = np.zeros((size, size))

# Set the corner values (you can experiment with different values)
terrain[0, 0] = 0
terrain[0, size - 1] = 0
terrain[size - 1, 0] = 0
terrain[size - 1, size - 1] = 0

# Apply the midpoint displacement algorithm
midpoint_2d(terrain, 0, 0, n, rand_scale=1.0)

# Create a simple 2D plot with matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(terrain, cmap="terrain")
plt.colorbar(label="Elevation")
plt.title("Terrain Generated with Diamond-Square Algorithm")
plt.show()


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
