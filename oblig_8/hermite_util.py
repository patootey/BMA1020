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

    return -1  # default value if no interval is found


def hermite(s, t, x, v):
    # import list of time steps t, list of positions x
    # and list of speeds v
    # returns x(s), v(s) by Hermite interpolation
    i = closest(s, t)
    if i == -1:
        return 0, 0
    else:
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        v0, v1 = v[i], v[i + 1]

        # Normalize the parameter
        h = t1 - t0
        u = (s - t0) / h

        # Cubic Hermite basis functions
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2

        pos = h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1

        dh00 = (6 * u**2 - 6 * u) / h
        dh10 = 3 * u**2 - 4 * u + 1
        dh01 = (-6 * u**2 + 6 * u) / h
        dh11 = 3 * u**2 - 2 * u

        vel = dh00 * x0 + dh10 * v0 + dh01 * x1 + dh11 * v1

        return pos, vel


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

# Apply the interpolation for all points in tspan
all_pos = np.zeros(tspan.shape[0])
all_vel = np.zeros(tspan.shape[0])

for j, s in enumerate(tspan):
    all_pos[j], all_vel[j] = hermite(s, timedata, positiondata, velocitydata)


t = np.linspace(0, 6, 10)
ss = [-6, 2, 7, 1.5, 8, 0, 6, 12]
for s in ss:
    print(f"s = {s} is in sub-interval {closest(s, t)}.")

print(np.round(all_pos[0:10], 5))
print(np.round(all_vel[0:10], 5))
