import numpy as np

# a)


def integrate_ex1(f, start, stop, dt):
    t_values = np.arange(start, stop, dt)
    return np.sum(np.vectorize(f)(t_values) * dt)


# b)


def integrate_ex2(f, start, stop, dt):
    t_values = np.arange(start, stop, dt)
    f_values = np.vectorize(f)(t_values)
    return np.sum((f_values[:-1] + f_values[1:]) * dt / 2)


# c)


def integrate_ex3(f, start, stop, dt):
    t_values = np.arange(start, stop, dt)
    f_values = np.vectorize(f)(t_values)
    return (
        dt
        / 3
        * (
            f_values[0]
            + f_values[-1]
            + 4 * np.sum(f_values[1:-1:2])
            + 2 * np.sum(f_values[2:-1:2])
        )
    )
