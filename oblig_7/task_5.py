import numpy as np
from quaternions_extra import Quaternion, interpolate_quaternion


# (A) Right hand side in Eulers equation of rigid body motion
def euler(omega, torque, inertia):
    omega_cross_I_omega = np.cross(omega, np.dot(inertia, omega))
    # Solve the linear system I⋅ω̇ = M - ω×(Iω) for ω̇
    right_side = torque - omega_cross_I_omega
    omega_dot = np.linalg.solve(inertia, right_side)
    return omega_dot


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
for i in np.arange(1, n):
    # A quaternion based on the angular velocity
    omega_vector = angle_velocities[:, i - 1].reshape(
        3, 1
    )  # Reshape to make it a column vector
    omega = Quaternion([0], omega_vector)

    # Euler-update of the rotations using q̇ = 0.5 * (ω ⊙ q)
    q = q.add(omega.multiply(q).scale(0.5 * h))

    # Saving the results in a matrix
    Q[0, i] = q.scalar
    Q[1:, i] = q.vec[:, 0]

# Build a collection of quaternions based on the result of explicit Euler integration
Rotations = Quaternion(Q[0, :], Q[1:, :])

# (D) Find the rotation quaternion at a given time via interpolation
t = 4.75  # time instance of interest

# Find indices where tspan[idx] <= t < tspan[idx+1]
idx = np.where(tspan <= t)[0][-1]
if idx == n - 1:  # t is exactly the last time point
    Qout = Rotations.get_subset([idx])
else:
    t1, t2 = tspan[idx], tspan[idx + 1]
    q1, q2 = Rotations.get_subset([idx]), Rotations.get_subset([idx + 1])
    # Calculate interpolation parameter
    s = (t - t1) / (t2 - t1)
    Qout = interpolate_quaternion(q1, q2, s)
