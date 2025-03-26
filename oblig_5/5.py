import numpy as np

b = 0.42
m = 70
g = 9.81
h = 1
tVerdier = np.arange(0, 31, h)

vk = np.zeros(len(tVerdier))
vk[0] = 0

for i in range(len(tVerdier) - 1):
    vk[i + 1] = vk[i] + h * (-(b / m) * vk[i] * abs(vk[i]) - g)

xk = np.zeros(len(tVerdier))
xk[0] = 3000

for i in range(len(tVerdier) - 1):
    xk[i + 1] = xk[i] + (h / 2) * (vk[i] + vk[i + 1])
