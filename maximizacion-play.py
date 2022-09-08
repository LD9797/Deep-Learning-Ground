import torch
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

# https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python

MU_START = 10
MU_END = 50
SIGMA_START = 1.1
SIGMA_END = 2.2


#  N observations, K parameters = 2
def generate_data(n_observations: int, k_parameters=2):
    gaussian_distributions = []
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(MU_START, MU_END))
        sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
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


#  Creates a sample of a normal distribution
def sample_normal(n_observations: int):
    mu = torch.tensor(random.uniform(MU_START, MU_END))
    sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
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


def init_random_parameters(k_parameters=2):
    p_matrix = []
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(MU_START, MU_END))
        sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
        p_matrix.append([mu, sigma])
    p_matrix = torch.tensor(p_matrix)
    return p_matrix


init_random_parameters()
