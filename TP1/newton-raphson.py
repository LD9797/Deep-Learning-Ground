import math
import matplotlib.pyplot as plt
import sympy.core
import torch
from gdm import f, f_prima_y, f_prima_x
from sympy import diff, symbols, parse_expr, E, sympify, latex
import numpy as np


def function(expression: str = ""):
    if not expression:
        x = symbols('x')
        y = symbols('y')
        return x * E ** (-x**2 - y**2)
    return parse_expr(expression)


def hessian_matrix(expression: sympy.core.Expr, variables: list = None):
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
        hess_matrix[:, matrix_col_row] = column
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


h_matrix = hessian_matrix(function())
latex_h_matrix = matrix_to_latex(h_matrix)
pass
