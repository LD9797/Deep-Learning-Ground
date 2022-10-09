import math
import matplotlib.pyplot as plt
import torch


def f(x, y):
    return x * math.e ** (-x**2 - y**2)


def f_prima_x(x, y):
    return (1 - 2 * x**2) * math.e ** (-x**2 -y**2)


def f_prima_y(x, y):
    return -2 * x * y * math.e ** (-y**2 - x**2)


def gradient_descent_momentum(initial_position, derivative_x, derivative_y, epochs=5, momentum=0.1, alpha=0.05):
    agent = initial_position
    agents = [agent]
    inertia = 0
    for epoc in range(epochs):
        gradient = torch.Tensor([derivative_x(agent[0], agent[1]), derivative_y(agent[0], agent[1])])
        agent = agent - ((momentum * inertia) + alpha * gradient)
        agents.append(agent)
        inertia = (momentum * inertia) + alpha * (1 - momentum) * gradient
    return agents


if __name__ == "__main__":
    init_position = torch.Tensor([0.5, -0.23])
    thetas = gradient_descent_momentum(init_position, f_prima_x, f_prima_y, epochs=10, alpha=0.25, momentum=0.5)

    #  Plot
    linspace_x = torch.linspace(-2, 2, steps=30)
    linspace_y = torch.linspace(-2, 2, steps=30)
    X, Y = torch.meshgrid(linspace_x, linspace_y, indexing="xy")
    Z = f(X, Y)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 2, 1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)  # Add a color bar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    thetas = torch.stack(thetas)
    agents_x = thetas[:, 0]
    agents_y = thetas[:, 1]
    ax.scatter(agents_x, agents_y, s=40, lw=0, color='red')
    plt.quiver(agents_x[:-1], agents_y[:-1], agents_x[1:]-agents_x[:-1], agents_y[1:]-agents_y[:-1],
               scale_units='xy', angles='xy', scale=1)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    agents_z = f(agents_x, agents_y)
    ax.scatter(agents_x, agents_y, agents_z, s=40, lw=0, color='red', alpha=1)
    ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.quiver(agents_x[:-1], agents_y[:-1], agents_z[:-1], (agents_x[1:]-agents_x[:-1]), (agents_y[1:]-agents_y[:-1]),
              (agents_z[1:]-agents_z[:-1]), length=1)

    plt.show()
