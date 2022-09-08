import torch
import matplotlib.pyplot as plt
import random
from scipy.stats import norm


def plot_bar(bins, histogram):
    fig = plt.figure()
    axes = fig.add_axes([0, 0, 1, 1])
    axes.bar(bins[1:].tolist(), histogram.tolist())


#  Segun el enunciado N oberbaciones y 2 funciones de densidad gauseannas
def generate_data(n_observations, k_parameters=2):
    gaussian_distributions = []
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(10, 50))
        sigma = torch.tensor(random.uniform(1.1, 2.2))
        normal_dist = torch.distributions.Normal(mu, sigma)
        sample = normal_dist.sample((n_observations, 1)).squeeze()
        gaussian_distributions.append(sample)
    for distribution in gaussian_distributions:
        mean = torch.mean(distribution)
        var = torch.var(distribution)
        x_axis = torch.arange(min(distribution) - 5, max(distribution) + 5, 0.01)
        def randomize(): return random.randint(0, 255)
        color = '#%02X%02X%02X' % (randomize(), randomize(), randomize())
        plt.scatter(distribution.numpy(), torch.zeros(n_observations), s=1, c=color, alpha=0.5)
        plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mean.numpy(), var.numpy()), c=color,
                 label=r'$\mu=' + str(round(mean.item(), 2)) + r',\ \sigma=' + str(round(var.item(), 2)) + r'$')
    plt.legend()
    plt.show()
    return gaussian_distributions


#  generate_data(200)


def sample_normal(n_observations):
    mu = torch.tensor(random.uniform(10, 50))
    sigma = torch.tensor(random.uniform(1.1, 2.2))
    normal_dist = torch.distributions.Normal(mu, sigma)
    sample = normal_dist.sample((n_observations, 1)).squeeze()
    mean = torch.mean(sample)
    var = torch.var(sample)
    x_axis = torch.arange(min(sample) - 5, max(sample) + 5, 0.01)
    plt.scatter(sample.numpy(), torch.zeros(n_observations), s=1, c='blue', alpha=0.5)
    plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mean.numpy(), var.numpy()),
             label=r'$\mu=' + str(round(mean.item(), 2)) + r',\ \sigma=' + str(round(var.item(), 2)) + r'$')
    plt.legend()
    plt.show()


generate_data(200)

# https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python
