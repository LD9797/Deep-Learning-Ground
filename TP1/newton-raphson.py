import math
import matplotlib.pyplot as plt
import sympy.core
import torch
from gdm import f, f_prima_y, f_prima_x
from sympy import diff, symbols, parse_expr, E, sympify, latex
import numpy as np
from torch.autograd import Variable


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
    for variable in variables:
        print("Building column #" + str(matrix_col_row + 1) + " and row #" + str(matrix_col_row + 1))
        first_derivative = sympify(diff(expression, variable))
        print("df/d" + variable + "=" + str(first_derivative))
        variable_index = variables.index(variable)
        column = []
        for second_variable in variables[variable_index:]:
            second_derivative = sympify(diff(first_derivative, second_variable))
            print("df/d" + second_variable + "d" + variable + "=" + str(second_derivative))
            column.append(second_derivative)
        hess_matrix[matrix_col_row:, matrix_col_row] = column
        row = []
        for second_variable in variables[variable_index+1:]:
            derivative_second_var = sympify(diff(expression, second_variable))
            print("df/d" + second_variable + "=" + str(derivative_second_var))
            second_derivative = sympify(diff(derivative_second_var, variable))
            print("df/d" + variable + "d" + second_variable + "=" + str(second_derivative))
            row.append(second_derivative)
        hess_matrix[matrix_col_row, matrix_col_row + 1:] = row
        matrix_col_row += 1
    return hess_matrix


def matrix_to_latex(matrix):
    latex_matrix = r'\begin{pmatrix}'
    for row in matrix:
        element_latex = ""
        for element in row:
            element_latex += latex(element) + " & "
        element_latex = element_latex[:len(element_latex) - 3] + r'\\'
        latex_matrix += element_latex
    latex_matrix += r'\end{pmatrix}'
    return latex_matrix


def torch_hessian_matrix(point: tuple):
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


def newton_raphson(initial_position, derivative_x, derivative_y, epochs=5):
    agent = initial_position
    agents = [agent]
    for epoc in range(epochs):
        gradient = torch.Tensor([derivative_x(agent[0], agent[1]), derivative_y(agent[0], agent[1])])
        gradient = gradient.resize_(2, 1)
        hessian_matrix = torch_hessian_matrix((agent[0], agent[1]))
        agent.resize_(2, 1)
        agent = agent + (torch.mm(-torch.inverse(hessian_matrix), gradient))
        agents.append(agent)
    for agent in agents:
        agent.resize_(2)
    return agents


h_matrix = visual_hessian_matrix(visual_function())
#  latex_h_matrix = matrix_to_latex(h_matrix)
#  torch_hessian = torch_hessian_matrix((2, 3))


if __name__ == "__main__":
    init_position = torch.Tensor([0.5, -0.23])
    thetas = newton_raphson(init_position, f_prima_x, f_prima_y)

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

