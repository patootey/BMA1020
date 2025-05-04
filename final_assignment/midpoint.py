"""
Code that generates a height map.

@Author Mathilde Borsodi Hegna
"""

import numpy as np


def midpoint_2d(
    heightmap: np.ndarray,
    i: int,
    j: int,
    grid_n: int,
    rand_scale: float,
    terrain_roughness: float = 1.0,
) -> None:
    """
    Generate a heightmap recursivly.

    :param heightmap:
    :param i: Start point.
    :param j: Start point.
    :param grid_n: The power of the grid.
    :param rand_scale: The scale.
    :param terrain_roughness: How rought the terrain is.
    """
    rng = np.random.default_rng(12345)
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

    current_scale = rand_scale * terrain_roughness
    mid = np.mean(corners) + current_scale * (rng.random() - 0.5)
    heightmap[center_i, center_j] = mid

    # Only fill in sides if they are empty
    # Left side
    if np.isnan(heightmap[center_i, j]):
        heightmap[center_i, j] = np.mean([top_left, top_right, mid]) + current_scale * (
            rng.random() - 0.5
        )
    # Bottom side
    if np.isnan(heightmap[i + last_index, center_j]):
        heightmap[i + last_index, center_j] = np.mean(
            [top_right, bottom_right, mid]
        ) + current_scale * (rng.random() - 0.5)
    # Right side
    if np.isnan(heightmap[center_i, j + last_index]):
        heightmap[center_i, j + last_index] = np.mean(
            [bottom_left, bottom_right, mid]
        ) + current_scale * (rng.random() - 0.5)
    # Top side
    if np.isnan(heightmap[i, center_j]):
        heightmap[i, center_j] = np.mean(
            [top_left, bottom_left, mid]
        ) + current_scale * (rng.random() - 0.5)

    # Recursively apply to the four quadrants
    if grid_n > 1:
        # Decrease scale for lower levels
        rand_scale_max = max(rand_scale * 0.6, 0.2)
        midpoint_2d(
            heightmap, i, j, (grid_n - 1), rand_scale_max, terrain_roughness
        )  # Top-left
        midpoint_2d(
            heightmap, i, j + half, (grid_n - 1), rand_scale_max, terrain_roughness
        )  # Top-right
        midpoint_2d(
            heightmap, i + half, j, (grid_n - 1), rand_scale_max, terrain_roughness
        )  # Bottom-left
        midpoint_2d(
            heightmap,
            i + half,
            j + half,
            (grid_n - 1),
            rand_scale_max,
            terrain_roughness,
        )  # Bottom-right
    return
