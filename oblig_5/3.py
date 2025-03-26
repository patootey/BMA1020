import numpy as np


def f(t, u):
    x, y = u
    dxdt = y
    dydt = 2 * x + np.sin(t) - y
    return np.array([dxdt, dydt])


def eulerMetode(f, u0, t0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    u = np.zeros((len(t), len(u0)))
    u[0] = u0

    for i in range(1, len(t)):
        u[i] = u[i - 1] + h * f(t[i - 1], u[i - 1])

    return t, u


u0 = np.array([1, 0])
t0 = 0
tEnd = 1
h = 0.1

t, u = eulerMetode(f, u0, t0, tEnd, h)

x = u[-1, 0]
y = u[-1, 1]
