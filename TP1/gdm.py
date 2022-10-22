import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad


def f(x, y):
    return x * math.e ** (-x**2 - y**2)


def gradient_descent_momentum(initial_position, epochs=5, momentum=0.1, alpha=0.05, epsilon=0.2):
    agent = initial_position
    agent.requires_grad = True
    agents = [agent]
    inertia = 0
    for epoc in range(epochs):
        function_eval = f(agent[:1], agent[1:])
        gradient = grad(function_eval, agent, create_graph=True)[0]
        if torch.norm(gradient) < epsilon:  # Condition over the gradient
            break
        agent = agent - ((momentum * inertia) + alpha * gradient)
        theta = agent.detach()
        agents.append(theta)
        inertia = (momentum * inertia) + alpha * (1 - momentum) * gradient
    agents[0] = agents[0].detach()
    return agents


if __name__ == "__main__":
    init_position = torch.Tensor([0.5, -0.23])
    thetas = gradient_descent_momentum(init_position, epochs=10, alpha=0.25, momentum=0.5)

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
    #  ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.quiver(agents_x[:-1], agents_y[:-1], agents_z[:-1], (agents_x[1:]-agents_x[:-1]), (agents_y[1:]-agents_y[:-1]),
              (agents_z[1:]-agents_z[:-1]), length=1)

    plt.show()
