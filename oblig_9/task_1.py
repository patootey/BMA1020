import numpy as np

rng = np.random.default_rng(12345)


def midpoint_2d_no_extra(A, i, j, square_size: float, rand_scale) -> None:
    size = 2**square_size

    if square_size < 1:
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
    if square_size > 1:
        midpoint_2d(A, i, j, square_size - 1, rand_scale)  # Top-left
        midpoint_2d(A, i, j + half, square_size - 1, rand_scale)  # Top-right
        midpoint_2d(A, i + half, j, square_size - 1, rand_scale)  # Bottom-left
        midpoint_2d(A, i + half, j + half, square_size - 1, rand_scale)  # Bottom-right

    return A[i : i + size + 1, j : j + size + 1]


def midpoint_2d(heightmap, i, j, square_size: float, rand_scale) -> None:
    # Calculate the side length between points in current square
    size = 2**square_size

    if square_size < 1:
        return heightmap[i : i + size + 1, j : j + size + 1]

    half = size // 2

    # DIAMOND STEP
    # Calculate midpoint of the square
    mid_i = i + half
    mid_j = j + half

    top_left = heightmap[i, j]
    top_right = heightmap[i, j + size]
    bottom_left = heightmap[i + size, j]
    bottom_right = heightmap[i + size, j + size]
    corners = np.array([top_left, top_right, bottom_left, bottom_right])

    heightmap[mid_i, mid_j] = np.mean(corners) + rand_scale * size * (
        rng.random() - 0.5
    )

    # skal jeg ha med mid?
    # Top side
    heightmap[i, mid_j] = np.mean(
        [top_left, top_right, heightmap[mid_i, mid_j]]
    ) + rand_scale * size * (rng.random() - 0.5)
    # Right side
    heightmap[mid_i, j + size] = np.mean(
        [top_right, bottom_right, heightmap[mid_i, mid_j]]
    ) + rand_scale * size * (rng.random() - 0.5)
    # Bottom side
    heightmap[i + size, mid_j] = np.mean(
        [bottom_left, bottom_right, heightmap[mid_i, mid_j]]
    ) + rand_scale * size * (rng.random() - 0.5)
    # Left side
    heightmap[mid_i, j] = np.mean(
        [top_left, bottom_left, heightmap[mid_i, mid_j]]
    ) + rand_scale * size * (rng.random() - 0.5)

    # Recursively apply to the four squares
    if square_size > 1:
        midpoint_2d(heightmap, i, j, square_size - 1, rand_scale)  # Top-left
        midpoint_2d(heightmap, i, j + half, square_size - 1, rand_scale)  # Top-right
        midpoint_2d(heightmap, i + half, j, square_size - 1, rand_scale)  # Bottom-left
        midpoint_2d(
            heightmap, i + half, j + half, square_size - 1, rand_scale
        )  # Bottom-right

    return heightmap[i : i + size + 1, j : j + size + 1]
