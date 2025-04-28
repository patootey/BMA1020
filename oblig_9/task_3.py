# Generate n unit vectors with 'Lambertian' distribution
# from a triangular element

# and n points in the triangle
import numpy as np

def random_rays(rng, x0, x1, x2, n):
    # rng generator; x0,x1,x2 vertices of triangle; n number of rays
    # first find the normal vectors
    u = x1 - x0
    v = x2 - x0
    N = np.cross(u, v)
    n = np.linalg.norm(N)
    N = N / n
    # now an orthonormal basis for surface, with third component the normal component
    u1 = u / np.linalg.norm(u)
    v1 = np.cross(N, u1)
    B = np.array([u1, v1, normal])
    # generate n random polar and azimuthal angles
    theta = np.pi * rng.random(size=n) / 2
    phi = 2 * np.pi * rng.random(size=n)
    # transform to 3d coordinates in the standard basis
    r = B @ np.array(
        [np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)]
    )
    # Now generate n points inside the triangle
    # this bit you can do yourself!
    x = ...
    # Return the values
    return x, r


def form_factors_raytrace(nodes,elements,n,rng)
    n = np.size(elements, axis=0)
    F = np.zeros((n,n))
    for i, el in enumerate(elements):
        # generate n rays from random points x in element i, in random directions r 
        x, r = random_rays(rng, nodes[el][0], nodes[el][1], nodes[el][2], n)
        # find out which element (if any) each ray hits
        # add 1/n to F[i,j] for each ray from i hitting j
    return F