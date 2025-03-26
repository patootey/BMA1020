#
# u, v are lists of the form [a,b,c].
#

import math as m

u, v = [5, -1, 1], [3, 3, 2]


# the cross product of u and v
def cross(u, v):
    result = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]

    return result


print(cross(v, u))


# the unit vector in the same direction of u
def normalise(u):
    magnitude = m.sqrt(sum(x**2 for x in u))
    return [x / magnitude for x in u]


# the dot product of u and v
def dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


# projection of u onto v
def project(u, v):
    magnitude_squared = sum(x**2 for x in v)

    scalar_projection = dot(u, v) / magnitude_squared

    return [scalar_projection * x for x in v]


print(project(u, v))
