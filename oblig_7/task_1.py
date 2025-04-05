import numpy as np


# (A)
def divdiff(X):
    # X ... a list of points [xi, yi]
    if np.size(X, axis=0) == 1:
        return X[0][1]
    else:
        return (divdiff(X[1:]) - divdiff(X[:-1])) / (X[-1][0] - X[0][0])


# (B)
def newton(X, x):
    # intput
    # - X ... a list of points [xi, yi]
    # - x ... value (argument of function)
    # output: function value of the interpolating polynomial
    svar = 0
    for i in range(len(X)):
        c = divdiff(X[: i + 1])
        a = 1
        for k in range(i):
            a *= x - X[k][0]
        svar += c * a
    return svar


# A
ca = np.array([[1, 2], [3, 1], [5, 6], [7, 2]])
xa = np.linspace(0, 6, 100)
for i in range(1, 5):
    print(np.round(divdiff(ca[:i, :]), 5))

# B

cb = np.array([[1, 2], [3, 1], [5, 6]])
xb = np.linspace(0, 6, 50)
yb = np.zeros(50)
for i in np.arange(50):
    yb[i] = newton(cb, xb[i])
print(np.round(yb, 6))
