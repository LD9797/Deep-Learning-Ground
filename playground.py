import math
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


def derivada_funcion_a(x, y):
    return x / math.sqrt(x**2 + y**2)


def evaluar_grandiente(punto_0, punto_1, funcion_derivada, arrange):
    #  Arrange es de que valor a que valor se va a evaluar la funcion para graficarla
    xs = torch.linspace(arrange[0], arrange[1], steps=256)  # x1
    ys = torch.linspace(arrange[0], arrange[1], steps=256)  # x2
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    z = torch.sqrt(x**2 + y**2)
    punto_x, punto_y = punto_0[0], punto_0[1]
    # X, Y, Z
    vector_grandiente_punto_0 = [funcion_derivada(punto_x, punto_y), funcion_derivada(punto_y, punto_x), 0]
    punto_x, punto_y = punto_1[0], punto_1[1]
    vector_grandiente_punto_1 = [funcion_derivada(punto_x, punto_y), funcion_derivada(punto_y, punto_x), 0]
    inicio = [0, 0, 0]
    ax = plt.axes(projection='3d')
    #  Se dibujan los vectores
    ax.quiver(inicio[0], inicio[1], inicio[2], vector_grandiente_punto_0[0], vector_grandiente_punto_0[1],
              vector_grandiente_punto_0[2])
    ax.quiver(inicio[0], inicio[1], inicio[2], vector_grandiente_punto_1[0], vector_grandiente_punto_1[1],
              vector_grandiente_punto_1[2])
    #  Se dibuja el grafico 3D de la funcion
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #  Se muestra el grafico
    plt.show()


if __name__ == "__main__":
    evaluar_grandiente([5.2, 6.4], [5.2, 2.3], derivada_funcion_a, [-1, 1])
