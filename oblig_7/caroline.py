import numpy as np


def closest(s, t):
    # input: (ordered) list of (unique) time instances t
    # point where function to be evaluated s
    # output: index of sub interval

    # Check if s is within the range of t
    if s < t[0] or s > t[-1]:
        return -1

    # Find the index i such that t[i] <= s < t[i+1]
    for i in range(len(t) - 1):
        if t[i] <= s < t[i + 1]:
            return i
    return -1  # default value


def hermite(s, t, x, v):
    # import list of time steps t, list of positions x
    # and list of speeds v
    # returns x(s), v(s) by Hermite interpolation
    i = closest(s, t)
    if i == -1:
        return 0, 0
    # Code below this line to be adapted ---------------------------------------
    else:
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        v0, v1 = v[i], v[i + 1]

        # Normalize s to the interval [0, 1]
        h = t1 - t0
        tau = (s - t0) / h

        # Hermite basis functions with slight modification
        h00 = 2 * tau**3 - 3 * tau**2 + 1
        h10 = tau**3 - 2 * tau**2 + tau
        h01 = -2 * tau**3 + 3 * tau**2
        h11 = tau**3 - tau**2

        # Interpolate position with adjusted scaling
        x_s = h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1

        # Interpolate velocity with adjusted derivative
        v_s = (
            (6 * tau**2 - 6 * tau) / h * x0
            + (3 * tau**2 - 4 * tau + 1) * v0
            + (-6 * tau**2 + 6 * tau) / h * x1
            + (3 * tau**2 - 2 * tau) * v1
        )

        return x_s, v_s


timedata = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
positiondata = np.array(
    [
        3000.0,
        2969.34375,
        2888.65266797,
        2787.9685669,
        2686.79796028,
        2585.72790172,
        2484.63628808,
    ]
)
velocitydata = np.array(
    [0.0, -24.525, -40.02786563, -40.51941523, -40.41707007, -40.43897677, -40.43431414]
)
tspan = np.linspace(0, 17, 52)
all_pos, all_vel = np.zeros(tspan.shape[0]), np.zeros(tspan.shape[0])


for j, s in enumerate(tspan):
    all_pos[j], all_vel[j] = hermite(s, timedata, positiondata, velocitydata)


print(np.round(all_pos[0:10], 5))
print(np.round(all_vel[0:10], 5))
