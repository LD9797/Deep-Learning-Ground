import math
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


def f(x, y):
    return x * torch.e**(-x**2 - y**2)


x = torch.linspace(-6, 6, steps=30)
y = torch.linspace(-6, 6, steps=30)

X, Y = torch.meshgrid(x, y, indexing="xy")
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

print(f(math.sqrt(2)/2, 0))
