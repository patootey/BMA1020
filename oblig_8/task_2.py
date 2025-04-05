import numpy as np
### a)

# all arguments are 3d vectors (ie 1d numpy arrays with 3 elements)
def plane_intersect(X0,X1,X2,x,R):
    # First find the normal N
    N = ...
    # normal distance to the plane through X0, X1, X2
    d = ...
    # the intersection point with the plane through X0, X1, X2
    p = ...
    # find the angle of intersection (to the normal)
    theta = ...
    # must check if p is inside the triangle
    Inside = ... # True if p is inside, False otherwise
    # return 
    return Inside, p, N, theta

### b)

def snell(theta_i, n1, n2):
    # return False if there is no refracted ray (total internal reflection)
    # otherwise return the relevant angle
    # theta_i is the angle the incident ray makes with the normal to the surface
    # n1 is the index of refraction in the medium in front of the element
    # (this will usually be the value for air, i.e. nearly 1)
    # n2 is the index of refraction in the medium behind/inside the element
    # (this will usually be greater than 1)
    return 

# arguments X0, X1, X2, x, r are 3d vectors, as in a)
# n1, n2 and pol are scalars

def incident(X0, X1, X2, x, r, n2, n1 = 1, pol = 0.5):
    # refractive indices of materials
    # assume n1 = air/vacuum unless otherwise
    intersect, p, n, theta_i = plane_intersect(X0,X1,X2,x,r)
    if not(intersect):
        return False
    theta_t = snell(theta_i, n1, n2)
    if theta_t:
        ...
    # The wave impedances
    Z0 = 376 
    Z1 = Z0 / n1
    Z2 = Z0 / n2
    Rp = ...
    Rs = ...
    R = pol*Rp + (1-pol)*Rs
    # 1 - R is the power transmitted in the normal direction
    T = # we want the power of the refracted ray, which is closely related to 1 - R
    return R, np.pi - theta_i, T, theta_t