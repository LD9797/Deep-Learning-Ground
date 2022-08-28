import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import entropy


def plot_bar(bins, histogram):
    fig = plt.figure()
    axes = fig.add_axes([0, 0, 1, 1])
    axes.bar(bins[1:].tolist(), histogram.tolist())


#  Segun el enunciado N oberbaciones y 2 funciones de densidad gauseannas
def generate_data(n_obervations, k_parameters=2):
    gaussian_distributions = []
    for k in range(k_parameters):
        #  Genero miu aleatorio
        random_mean = torch.tensor(random.uniform(-10, 10))
        #  Genero sigma aleatorio
        random_scale = torch.tensor(random.uniform(5, 10))
        #  Genero distribucion con dichos parametros como salia en el jupyter del profe
        normal_dist = torch.distributions.Normal(random_mean, random_scale)
        gaussian_sample = normal_dist.sample((n_obervations, 1)).squeeze()
        #  Agrego la distribucion en el arreglo
        gaussian_distributions.append(gaussian_sample)

    # SOLO PARA GRAFICAR (Igual que en el jupyter del profe)
    fig = plt.figure(frameon=True, edgecolor='black')
    axes = fig.add_axes([0, 0, 1, 1])
    dist_number = 0
    for distribution in gaussian_distributions:
        histogram, bins = np.histogram(distribution.numpy(), bins=100, range=(-30, 30))
        p_1 = torch.tensor(histogram / histogram.sum())
        axes.bar(bins[1:].tolist(), p_1.tolist(), 1, label='Distribution #' + str(dist_number))
        dist_number += 1
    axes.legend()
    axes.grid(visible=True)

    # Aqui genero las N obervaciones aleatorias
    # Son basicamente un monton de numeros entre -30 y 30
    x = np.random.uniform(-30, 30, [n_obervations])
    y = np.zeros(n_obervations)
    # Aqui se grafican dichas obervaciones
    area = 40  # 0 to 15 point radii
    x2 = np.random.uniform(10, 60, [n_obervations])
    axes.scatter(x, y, s=area, c='orange', alpha=0.5)
    axes.scatter(x2, y, s=area, c='blue', alpha=0.5)
    plt.show()


generate_data(200)

# Esto es lo que venia en el jupiter
# n = 2000
#
# # Create gaussian noise values
# # Params Media y Desviacion estandar
# normal_dist = torch.distributions.Normal(torch.tensor([10.0]), torch.tensor([1.02]))
# gaussian_sample = normal_dist.sample((n, 1)).squeeze()
# print(gaussian_sample)
#
# histogram, bins = np.histogram(gaussian_sample.numpy(), bins=100, range=(0, 20))
# p_1 = torch.tensor(histogram / histogram.sum())
#
# # Plot histogram
# plot_bar(bins, p_1.numpy())
# print("verify p[x] property")
# print("Sum values is 1: ", p_1.sum())
#
# mean_x = torch.mean(gaussian_sample)
# var_x = torch.var(gaussian_sample)
# print("mean_x ", mean_x)
# print("var_x ", var_x)
