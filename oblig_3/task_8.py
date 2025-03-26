import numpy as np

np.random.seed(5)
M = np.random.rand(20, 20)
M[M <= 0.8] = 0.0
M[M > 0.8] = 1.0

answer = np.where(M == 0, 20.0, 1.0)
for k in range(20):
    answer = np.minimum(answer, answer[:, k][:, np.newaxis] + answer[k])
