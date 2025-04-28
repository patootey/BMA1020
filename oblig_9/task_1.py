import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)


# takes an array (1d) A, and a start (i) and end point (j)
def midpoint(A, i, j, rand_scale):
    # stop if the distance is too small
    if j - i == 0:
        return A[i]
    if j - i == 1:
        return A[i : j + 1]
    # find the mid point
    mid = (i + j) // 2
    # take the average to set the midpoint, then add a random number
    A[mid] = (A[i] + A[j]) / 2 + rand_scale * (j - i) * (rng.random() - 0.5)
    # recursion on the first half
    A[i : mid + 1] = midpoint(A, i, mid, rand_scale)
    # recursion on the second half
    A[mid : j + 1] = midpoint(A, mid, j, rand_scale)
    return A[i : j + 1]


# takes in an array, with start point (i,j) and square size 2**n + 1
def midpoint_2d(A, i, j, n, rand_scale):
    # write your code here
    return  # write your code here
