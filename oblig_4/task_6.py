import numpy as np

# Define matrix A and vector b
A = np.array([[2, 4, 5, 0], [3, -1, -2, 9], [2, 6, 4, -10], [5, -1, 2, 23]])

b = np.array([-7, 53, -66, 131])

# QR Decomposition
Q, R = np.linalg.qr(A)

# Compute Q^T * b
b_new = Q.T @ b

# Extract components for solving the reduced system
R_1 = R[:3, :3]  # Upper-left 3x3 submatrix
r_rest = R[:3, 3]  # Last column (except last row)
b_rest = b_new[:3]  # First three elements of transformed b

# Solve for y_rest when y_4 = 5
y_n_5 = 5
y_rest_5 = np.linalg.solve(R_1, b_rest - y_n_5 * r_rest)
y_5 = np.hstack([y_rest_5, y_n_5])

# Solve for y_rest when y_4 = 6
y_n_6 = 6
y_rest_6 = np.linalg.solve(R_1, b_rest - y_n_6 * r_rest)
y_6 = np.hstack([y_rest_6, y_n_6])

print(y_5, y_6)
