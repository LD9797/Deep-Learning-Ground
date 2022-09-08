import torch
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm

# https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python
# https://jeltef.github.io/PyLaTeX/current/examples/full.html

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


#  Generates a k x 2 dimensions matrix with random mu and sigma
def init_random_parameters(k_parameters=2):
    p_matrix = []
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(MU_START, MU_END))
        sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
        p_matrix.append([mu, sigma])
    p_matrix = torch.tensor(p_matrix)
    return p_matrix


def calculate_likelihood_gaussian_observation(x_n, mu_k, sigma_k):
    def probability_density_function(x, mu, sigma):
        return (1/math.sqrt(2 * math.pi * sigma**2)) * math.e**(-(1/2) * ((x-mu) / sigma)**2)
    return probability_density_function(x_n, mu_k, sigma_k)


#  generate_data(n_observations: int, k_parameters=2) -> x_dataset: list 2x1
#  init_random_parameters(k_parameters=2) -> parameters_matrix: list 2x2
#  Returns:
#  likelihood_matrix = [ [1, 0],
#                        [0, 1],
#                        ...,
#                        [1,0]]
def calculate_membership_dataset(x_dataset, parameters_matrix):
    likelihood_matrix = []
    for dataset in x_dataset:
        for data in dataset:
            data_likelihood = []
            for matrix in parameters_matrix:
                mu = matrix[0]
                sigma = matrix[1]
                likelihood = calculate_likelihood_gaussian_observation(data.item(), mu.item(), sigma.item())
                data_likelihood.append(likelihood)
            for index in range(len(data_likelihood)):
                data_likelihood[index] = 0.0 if data_likelihood[index] != max(data_likelihood) else 1.0
            likelihood_matrix.append(data_likelihood)
    likelihood_matrix = torch.tensor(likelihood_matrix)
    return likelihood_matrix


#  calculate_membership_dataset(x_dataset, parameters_matrix) -> membership_data
def recalculate_parameters(membership_data):
    membership_data = torch.transpose(membership_data, 0, 1)
    new_parameters = []
    for k in membership_data:
        mu = torch.mean(k)
        sigma = torch.var(k)
        new_parameters.append([mu, sigma])
    return new_parameters


def expectation_maximization(observations=200, k_parameters=2, iterations=5):
    my_data = generate_data(observations, k_parameters)
    parameters = init_random_parameters(k_parameters)
    print("Initial parameters: " + str(parameters))
    for iteration in range(iterations):
        print("Iteration #" + str(iteration))
        membership_data = calculate_membership_dataset(my_data, parameters)
        print("Membership dataset: " + str(membership_data))
        parameters = recalculate_parameters(membership_data)
        print("New parameters: " + str(parameters))


expectation_maximization()
