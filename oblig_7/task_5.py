import numpy as np
from quaternions_extra import Quaternion, interpolate_quaternion


# (A) Right hand side in Eulers equation of rigid body motion
def euler(omega, torque, inertia):
    omega_cross = np.cross(omega, np.dot(inertia, omega))
    rhs = torque - omega_cross
    domega = np.linalg.solve(inertia, rhs)
    return domega


# Explicit Euler method for time integration, note that we allow the right-hand-side to
# depend on further parameters "args"
def euler_integrate(fun, y0, t0, tend, n, args=()):
    ndim = np.shape(y0)[0]
    out = np.zeros((ndim, n))
    out[:, 0] = y0
    tspan, h = np.linspace(t0, tend, n, retstep=True)
    for i in np.arange(1, n):
        out[:, i] = out[:, i - 1] + h * fun(out[:, i - 1], *args)
    return tspan, out


# Eulers method for the Euler equation of rigid body motion
t0, tend, n = 0, 10.0, 100
angle0 = np.array([1, 1, 1])  # initial angular velocity
torque = np.array([1.0, -2.0, 0.0])  # applied torque
Inertia_tensor = np.array(
    [
        [3.7e01, 1.9e-01, -2.5e00],
        [1.9e-01, 6.0e01, -4.5e-02],
        [-2.5e00, -4.5e-02, 6.6e01],
    ]
)
tspan, angle_velocities = euler_integrate(
    euler, angle0, t0, tend, n, (torque, Inertia_tensor)
)
h = tspan[1] - tspan[0]

# (C) Updating the rotations (quaternions), again using Eulers method, but for quaternion operation
q = Quaternion([1], [[0], [0], [0]])  # initial rotation
Q = np.zeros((4, n))
Q[0, 1] = q.scalar
Q[1:, 1] = q.vec[:, 0]

for i in np.arange(1, n):
    omega_vec = angle_velocities[:, i - 1].reshape(3, 1)
    omega_q = Quaternion([0], omega_vec)
    dq = omega_q * q
    dq.scalar *= 0.5
    dq.vec *= 0.5
    dq.scalar *= h
    dq.vec *= h
    q = q + dq
    Q[0, i] = q.scalar
    Q[1:, i] = q.vec[:, 0]

# Build a collection of quaternions based on the result of explicit Euler integration
Rotations = Quaternion(Q[0, :], Q[1:, :])

# (D) Find the rotation quaternion at a given time via interpolation
t = 4.75  # time instance of interest
Qout = interpolate_quaternion(Rotations, tspan, t)
