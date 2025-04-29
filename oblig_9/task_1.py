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

    # ALSO MIDPOINT
    # MISSING
    # numpy average
    # https://medium.com/@nickobrien/diamond-square-algorithm-explanation-and-c-implementation-5efa891e486f
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
