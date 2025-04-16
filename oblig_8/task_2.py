import numpy as np

# a)


# The functions from programming assignment 1
def vector_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# The functions from programming assignment 1
def line_intersect_plane(plane_p, plane_n, point_1, point_2):
    # Normalize the plane normal
    plane_n = vector_normalize(plane_n)

    # Plane equation D
    plane_d = -np.dot(plane_n, plane_p)
    ad = np.dot(point_1, plane_n)
    bd = np.dot(point_2, plane_n)

    if ad == bd:  # Line is parallel to the plane
        return None

    t = (-plane_d - ad) / (bd - ad)
    intersect = (point_2 - point_1) * t
    return point_1 + intersect


# all arguments are 3d vectors (ie 1d numpy arrays with 3 elements)
def plane_intersect(X0, X1, X2, x, R):
    # First find the normal N
    vector1 = X1 - X0
    vector2 = X2 - X0
    N = np.cross(vector1, vector2)
    if np.linalg.norm(N) == 0:
        return False, None, None, None  # Degenerate triangle
    N = vector_normalize(N)

    # må gjøre en sjekk om den er parallel
    # d = np.dot(vector1, vector2)
    #     # If line is parallel with the plane

    intersection = line_intersect_plane(X0, N, x, x + R)
    if intersection is None:
        return False, None, None, None

    p = intersection
    # find the angle of intersection (to the normal)
    R_normalized = vector_normalize(R)
    cos_theta = np.dot(-N, R_normalized)  # Cosine of angle with normal

    # If ray points away from triangle (wrong side)
    if cos_theta <= 0:
        return False, None, None, None

    theta = np.arccos(cos_theta)

    # must check if p is inside the triangle
    # Using barycentric coordinates to check if point is inside triangle
    v0 = X2 - X0
    v1 = X1 - X0
    v2 = p - X0

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return False, None, None, None

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # Check if point is in triangle
    Inside = (u >= 0) and (v >= 0) and (u + v <= 1)

    return Inside, p, N, theta


# b)


def snell(theta_i, n1, n2):
    # return False if there is no refracted ray (total internal reflection)
    # otherwise return the relevant angle
    # theta_i is the angle the incident ray makes with the normal to the surface
    # n1 is the index of refraction in the medium in front of the element
    # (this will usually be the value for air, i.e. nearly 1)
    # n2 is the index of refraction in the medium behind/inside the element
    # (this will usually be greater than 1)
    sin_theta_t = (n1 / n2) * np.sin(theta_i)

    if np.abs(sin_theta_t) > 1.0:
        # Total internal reflection
        return False

    theta_t = np.arcsin(sin_theta_t)
    return theta_t


# arguments X0, X1, X2, x, r are 3d vectors, as in a)
# n1, n2 and pol are scalars


def incident(X0, X1, X2, x, r, n2, n1=1, pol=0.5):
    # refractive indices of materials
    # assume n1 = air/vacuum unless otherwise
    intersect, p, n, theta_i = plane_intersect(X0, X1, X2, x, r)
    if not (intersect):
        return False
    theta_t = snell(theta_i, n1, n2)
    if theta_t:
        # Calculate cos for Fresnel equations
        cos_i = np.cos(theta_i)
        cos_t = np.cos(theta_t)
    # The wave impedances
    Z0 = 376
    Z1 = Z0 / n1
    Z2 = Z0 / n2

    if theta_t is False:  # Total internal reflection
        Rp = 1.0
        Rs = 1.0
    else:
        # Fresnel equations using the wave impedances
        # For p-polarization (parallel)
        Rp = ((Z2 * cos_i - Z1 * cos_t) / (Z2 * cos_i + Z1 * cos_t)) ** 2

        # For s-polarization (perpendicular)
        Rs = ((Z1 * cos_i - Z2 * cos_t) / (Z1 * cos_i + Z2 * cos_t)) ** 2

    R = pol * Rp + (1 - pol) * Rs
    # 1 - R is the power transmitted in the normal direction
    T = 1 - R

    return R, np.pi - theta_i, T, theta_t


X0 = np.array([0, 0, 1])
X2 = np.array([0, 1, 1])
X1 = np.array([1, 0, 1])
x = np.array([0, 0, 0])
r = np.array([0.1, 0, 1])
intersects, y1, y2, p = plane_intersect(X0, X1, X2, x, r)
print(intersects)
print(np.round(y1, decimals=6))
print(np.round(y2, decimals=6))
print(np.round(p, decimals=6))
