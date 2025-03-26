import numpy as np

T = 8
N = 100

h = T / N
tn = np.linspace(0, T, N + 1)

theta0 = np.pi / 2  # start vinkel
phi0 = 0  # start velocity

theta_i = np.zeros(N + 1)
phi_i = np.zeros(N + 1)

theta_i[0] = theta0
phi_i[0] = phi0

for n in range(N):
    phi_i[n + 1] = phi_i[n] - h * np.sin(theta_i[n])
    theta_i[n + 1] = theta_i[n] + h * phi_i[n + 1]
