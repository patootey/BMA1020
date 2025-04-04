import numpy as np


# (A)
def divdiff(X):
    # X ... a list of points [xi, yi]
    if len(X) == 1:
        return X[0][1]
    else:
        # Here recursive call of divdiff
        X_right = X[:-1]
        X_left = X[1:]
        return (divdiff(X_right) - divdiff(X_left)) / (X[-1][0] - X[0][0])


# (B)
def newton(X, x):
    # intput
    # - X ... a list of points [xi, yi]
    # - x ... value (argument of function)
    # output: function value of the interpolating polynomial
    n = len(X)
    result = X[0][1]  # Start with c0

    # For each term, calculate the product (x-x0)*(x-x1)*...*(x-x_{i-1})
    prod = 1.0

    for i in range(1, n):
        # Update the product term by multiplying with (x-x_{i-1})
        prod *= x - X[i - 1][0]

        # Calculate the coefficient using divdiff
        coef = divdiff(X[: i + 1])

        # Add this term to the result
        result += coef * prod

    return result


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
