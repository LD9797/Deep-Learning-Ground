import math
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np


def f_original(x, y):
    return x**2 - y**2


def f(x, y):
    return x * math.e ** (-x**2 -y**2)


def f_prima_x(x, y):
    return -(2*x**2 - 1) * math.e**(-x**2 -y**2)


def f_prima_x_original(x):
    return 2 * x


def f_prima_y(x, y):
    return -2 * x * y * math.e ** (-y**2 -x**2)


def f_prima_y_original(y):
    return -2 * y


def gradient_descent_momentum(initial_position, derivative_x, derivative_y, epochs=5, momentum=0.1, alpha=0.05):
    agents = []
    agent = initial_position
    agents.append(agent)
    inertia = 0
    for epoc in range(epochs):
        gradient = torch.Tensor([derivative_x(agent[0], agent[1]), derivative_y(agent[0], agent[1])])
        agent = agent - (momentum * inertia) + alpha * gradient
        agents.append(agent)
        inertia = (momentum * inertia) + (alpha * (1 - momentum)) * gradient
    return agents


init_position = torch.Tensor([0.5, 0.0])
thetas = gradient_descent_momentum(init_position, f_prima_x, f_prima_y)




def descenso_gradiente(pto_inicial : torch.Tensor, epocs, learning_rate):
    pto_inicial_z = torch.Tensor([f(pto_inicial[0], pto_inicial[1])])
    thetas = [torch.cat((pto_inicial, pto_inicial_z))]
    for epoc in range(epocs):
        theta = pto_inicial - learning_rate * \
                torch.Tensor([f_prima_x(pto_inicial[0], pto_inicial[1]), f_prima_y(pto_inicial[0], pto_inicial[1])])
        theta_z = torch.Tensor([f(theta[0], theta[1])])
        new_theta = torch.cat((theta, theta_z))
        thetas.append(new_theta)
        pto_inicial = theta
    return thetas


thetas = descenso_gradiente(torch.Tensor([0.5, 0.0]), 10, 0.05)

x = torch.linspace(-4, 4, steps=30)
y = torch.linspace(-4, 4, steps=30)

X, Y = torch.meshgrid(x, y, indexing="xy")
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

thetas = torch.stack(thetas)
x = thetas[:, 0]
y = thetas[:, 1]
z = thetas[:, 2]
z.apply_(lambda elem: (elem * 1))
ax.scatter(x, y, z, s=5, alpha=1, color='red')
ax.plot(x, y, z, color='black', linestyle='dashed')

plt.show()

print(f(math.sqrt(2)/2, 0))
