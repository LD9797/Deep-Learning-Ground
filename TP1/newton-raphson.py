import math
import matplotlib.pyplot as plt
import sympy.core
import torch
from gdm import f
from sympy import diff, symbols, parse_expr, E, sympify, latex
import numpy as np
from torch.autograd import Variable, grad
from enum import Enum


def visual_function(expression: str = ""):
    if not expression:
        x = symbols('x')
        y = symbols('y')
        return x * E ** (-x**2 - y**2)
    return parse_expr(expression)


def visual_hessian_matrix(expression: sympy.core.Expr, variables: list = None):
    if variables is None:
        variables = ["x", "y"]
    hess_matrix = np.empty((len(variables), len(variables)), dtype=sympy.core.Expr)
    matrix_col_row = 0
    derivatives = []
    for variable in variables:
        print("Building column #" + str(matrix_col_row + 1) + " and row #" + str(matrix_col_row + 1))
        first_derivative = sympify(diff(expression, variable))
        der = "df/d" + variable + "=" + str(first_derivative)
        derivatives.append(der)
        print(der)
        variable_index = variables.index(variable)
        column = []
        for second_variable in variables[variable_index:]:
            second_derivative = sympify(diff(first_derivative, second_variable))
            der = "df/d" + second_variable + "d" + variable + "=" + str(second_derivative)
            derivatives.append(der)
            print(der)
            column.append(second_derivative)
        hess_matrix[matrix_col_row:, matrix_col_row] = column
        row = []
        for second_variable in variables[variable_index+1:]:
            derivative_second_var = sympify(diff(expression, second_variable))
            der = "df/d" + second_variable + "=" + str(derivative_second_var)
            derivatives.append(der)
            print(der)
            second_derivative = sympify(diff(derivative_second_var, variable))
            der = "df/d" + variable + "d" + second_variable + "=" + str(second_derivative)
            derivatives.append(der)
            print(der)
            row.append(second_derivative)
        hess_matrix[matrix_col_row, matrix_col_row + 1:] = row
        matrix_col_row += 1
    return hess_matrix, derivatives


def derivatives_to_latex(derivatives):
    latex_derivatives = []
    for der in derivatives:
        start = "$" + der[0: der.find("=")]
        func = sympify(der[der.find("=") + 1:])
        latex_derivative = start + "=" + latex(func) + "$\n"
        latex_derivatives.append(latex_derivative)
    return latex_derivatives


def matrix_to_latex(matrix):
    latex_matrix = r'$H=\begin{pmatrix}'
    for row in matrix:
        element_latex = ""
        for element in row:
            element_latex += latex(element) + " & "
        element_latex = element_latex[:len(element_latex) - 3] + r'\\'
        latex_matrix += element_latex
    latex_matrix += r'\end{pmatrix}$'
    return latex_matrix


def hessian_matrix_old(point: tuple):
    x = torch.FloatTensor([point[0]])
    y = torch.FloatTensor([point[1]])
    x, y = Variable(x, requires_grad=True), Variable(y, requires_grad=True)
    variables = [x, y]
    hess_matrix = torch.empty((len(variables), len(variables)))
    matrix_col_row = 0
    variables_string = ["x", "y"]
    for variable in variables:
        expression = x * math.e ** (-x ** 2 - y ** 2)
        first_derivative = torch.autograd.grad(expression, variable, create_graph=True, retain_graph=True)
        print("df/d" + variables_string[variables.index(variable)] + "=" + str(first_derivative))
        variable_index = variables.index(variable)
        column = []
        for second_variable in variables[variable_index:]:
            second_derivative = torch.Tensor(torch.autograd.grad(first_derivative, second_variable, create_graph=True,
                                                                 retain_graph=True))
            print("df/d" + variables_string[variables.index(second_variable)] +
                  "d" + variables_string[variables.index(variable)] + "=" + str(second_derivative))
            column.append(second_derivative)
        hess_matrix[matrix_col_row:, matrix_col_row] = torch.Tensor(column)
        row = []
        for second_variable in variables[variable_index+1:]:
            derivative_second_var = torch.autograd.grad(expression, second_variable, create_graph=True,
                                                        retain_graph=True)
            print("df/d" + variables_string[variables.index(second_variable)] + "=" + str(derivative_second_var))
            second_derivative = torch.Tensor(torch.autograd.grad(derivative_second_var, variable))
            print("df/d" + variables_string[variables.index(variable)] +
                  "d" + variables_string[variables.index(second_variable)] + "=" + str(second_derivative))
            row.append(second_derivative)
        hess_matrix[matrix_col_row, matrix_col_row + 1:] = torch.Tensor(row)
        matrix_col_row += 1
    return hess_matrix


def newton_raphson_old(initial_position, derivative_x, derivative_y, epochs=5):
    agent = initial_position
    agents = [agent]
    for epoc in range(epochs):
        gradient = torch.Tensor([derivative_x(agent[0], agent[1]), derivative_y(agent[0], agent[1])])
        gradient = gradient.resize_(2, 1)
        hessian_matrix = hessian_matrix_old((agent[0], agent[1]))
        agent.resize_(2, 1)
        agent = agent + (torch.mm(-torch.inverse(hessian_matrix), gradient))
        agents.append(agent)
    for agent in agents:
        agent.resize_(2)
    return agents


def hessian_matrix(gradient, agent):
    print(f"First Derivative: {gradient}")
    dimensions = agent.shape[0]
    hess_matrix = torch.zeros(dimensions, dimensions)
    for dimension in range(dimensions):
        second_derivative = grad(gradient[dimension], agent, create_graph=True)[0]
        print(f"Second derivative on dimension {dimension}: {second_derivative}")
        hess_matrix[dimension:] = second_derivative
    print(f"Hessian Matrix: {hess_matrix}\n")
    return hess_matrix


class Point(Enum):
    LOCAL_MIN = "local minimum"
    LOCAL_MAX = "local maximum"
    SADDLE_POINT = "saddle point"
    NO_CONCLUSION = "no conclusion"


def point_status_by_determinant(hess_matrix):
    determinant = hess_matrix[0][0] * hess_matrix[1][1] - hess_matrix[0][1]**2
    if determinant > 0 and hess_matrix[0][0] > 0:
        return Point.LOCAL_MIN
    if determinant > 0 and hess_matrix[0][0] < 0:
        return Point.LOCAL_MAX
    if determinant < 0:
        return Point.SADDLE_POINT
    return Point.NO_CONCLUSION


def step_by_step_hessian():
    h_matrix, all_derivatives = visual_hessian_matrix(visual_function())
    all_derivatives = derivatives_to_latex(all_derivatives)
    latex_h_matrix = matrix_to_latex(h_matrix)
    print(f"Hessian Matrix: {h_matrix}" + "\n")
    print(f"Hessian Matrix in Latex: {latex_h_matrix}\n")
    return latex_h_matrix, all_derivatives


def newton_raphson(initial_position, function, epochs=5, damping_factor=0.4, convergence=-0.42, epsilon=0.2, run_with_fix=True, stop_at_saddle=False):  # To deactivate damping_factor use 1
    agent = initial_position
    agent.requires_grad = True
    agents = [agent]
    latex_h_matrix, all_derivatives = step_by_step_hessian()
    for epoch in range(epochs):
        print("---------epoch-" + str(epoch) + "-------------\n")
        print(f"Agent: {agent}\n")
        function_eval = function(agent[:1], agent[1:])
        gradient = grad(function_eval, agent, create_graph=True)[0]
        hess_matrix = hessian_matrix(gradient, agent)
        point_by_determinant = point_status_by_determinant(hess_matrix)
        print(f"The point ({round(agent[:1].item(), 3)}, {round(agent[1:].item(), 3)}) is: {point_by_determinant.value}\n")
        if stop_at_saddle and point_by_determinant == Point.SADDLE_POINT:
            print("Saddle point found ! Stopping...")
            break
        inverse_hess_matrix = torch.nan_to_num(torch.inverse(hess_matrix))
        hess_gradient = torch.mm(inverse_hess_matrix, gradient.view(gradient.shape[0], 1))
        new_agent = agent.view(agent.shape[0], 1) - damping_factor * hess_gradient
        new_agent = new_agent.view(agent.shape[0])
        if function(new_agent[:1], new_agent[1:]) > function_eval and run_with_fix:  # If it goes up then force it to go down
            agent = agent.view(agent.shape[0], 1) - damping_factor * torch.abs(hess_gradient)
            agent = agent.view(agent.shape[0])
        else:
            agent = new_agent
        theta = agent.detach()
        agents.append(theta)
        print(f"Gradient: {hess_gradient}")
        print(f"Inverse Hessian Matrix: {inverse_hess_matrix}\n")
        print(f"New agent: {theta}\n")
        if f(theta[:1], theta[1:]) <= convergence:
            print("Convergence reached ! ")
            break
        elif torch.norm(gradient) < epsilon:
            print("Tolerance reached !")
            break
    agents[0] = agents[0].detach()
    return agents, latex_h_matrix, all_derivatives


if __name__ == "__main__":
    init_position = torch.Tensor([0.5, 0.1])
    thetas, latex_hess_matrix, visual_derivatives = newton_raphson(init_position, f, epochs=6)
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
    plt.quiver(agents_x[:-1], agents_y[:-1], agents_x[1:] - agents_x[:-1], agents_y[1:] - agents_y[:-1],
               scale_units='xy', angles='xy', scale=1)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    agents_z = f(agents_x, agents_y)
    ax.scatter(agents_x, agents_y, agents_z, s=40, lw=0, color='red', alpha=1)
    #  ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.quiver(agents_x[:-1], agents_y[:-1], agents_z[:-1], (agents_x[1:] - agents_x[:-1]),
              (agents_y[1:] - agents_y[:-1]),
              (agents_z[1:] - agents_z[:-1]), length=1)

    plt.show()

