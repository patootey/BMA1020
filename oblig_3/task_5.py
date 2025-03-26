import numpy as np


#
# Answer to question a)
#
A1 = np.array([[3, 3, 1], [1, -3, 2], [1, 1, -5]])
b1 = np.array([1, 2, 3])
solA = np.linalg.solve(A1, b1)

# Answer to question b)
#
A2 = np.array([[3, 3, 1], [1, -3, 2], [1, 1, -5]])
b2 = np.array([8, 2, 6])
solB = np.linalg.solve(A2, b2)


#
# Answer to question c)
#
A3 = A2
A3inv = np.linalg.inv(A3)  # Compute the inverse of A
b3 = b2
solC = np.dot(A3inv, b3)  # Compute solution using A⁻¹ * b
