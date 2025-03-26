import numpy as np
from numpy.linalg import det as determinant


def matrix_minor(A, i, j):
    return np.delete(np.delete(A, i, axis=0), j, axis=1)


def laplace_det1(B, v, c):
    detB = determinant(B)  # Compute the determinant of B
    detA = detB * c  # Multiply by c
    return detA


def laplace_det2(A):
    m, n = A.shape
    if m != n:
        raise ValueError("Wrong dimensions of A")

    # Base case for 1x1 matrix
    if m == 1:
        return A[0, 0]

    detA = 0
    # Expand along the last row (index m-1)
    for j in range(n):
        # Compute the minor A_{(m-1), j} by removing the last row and j-th column
        minor = matrix_minor(A, m - 1, j)
        # Use (m-1)+j for the exponent to correctly reflect the 0-indexed row
        detA += (-1) ** ((m - 1) + j) * A[m - 1, j] * determinant(minor)

    return detA


# Example matrix A
A = np.array([[3, 4, -5], [1, 0, -4], [3, 2, 6]])

# Create the minors
A12 = matrix_minor(A, 0, 1)  # 1-2 minor
A33 = matrix_minor(A, 2, 2)  # 3-3 minor
