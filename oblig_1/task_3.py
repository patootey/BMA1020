#
# u, v are lists of the form [a,b,c].
#

import math as m

vector_1, vector_2 = [1, 5, 2], [4, 2, 1]
scalar = 3


# adds two vectors
def add(u, v):

    result = []
    for i in range(len(u)):
        result.append(u[i] + v[i])

    return result


print(add(vector_1, vector_2))


# scalar multiplication of scalar k and vector v
def mul(k, v):
    result = []
    for i in range(len(v)):
        result.append(v[i] * k)
    return result


print(mul(scalar, vector_2))


# the dot product of u and v
def dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


print(dot(vector_1, vector_2))


# the length of u
def vabs(u):
    return m.isqrt(sum([x**2 for x in u]))


print(vabs(vector_1))


for a, b in zip([5, 10], ["foo", "bar"]):
    print(a, b)

values = [1, 5, 10]
for a, b in zip(values, values[1:]):
    print(a, b)
