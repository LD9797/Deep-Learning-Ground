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


#  color = '#%02X%02X%02X' % (randomize(), randomize(), randomize())
def randomize(): return random.randint(0, 255)


def plot_observation(observation, show=False):
    mean = torch.mean(observation)
    var = torch.var(observation)
    x_axis = torch.arange(min(observation) - 5, max(observation) + 5, 0.01)
    plt.scatter(observation.numpy(), torch.zeros(len(observation)), s=1, alpha=0.5)
    plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mean.numpy(), var.numpy()),
             label=r'$\mu=' + str(round(mean.item(), 2)) + r',\ \sigma=' + str(round(var.item(), 2)) + r'$')
    if show:
        plt.legend()
        plt.show()


def plot_gaussian_distribution_and_observations(distribution_parameters, observations, show=False):
    for observation in observations:
        plot_observation(observation)
    param_number = 1
    for parameters in distribution_parameters:
        mean = parameters[0]
        var = parameters[1]
        x_axis = torch.arange(mean / 2, mean * 2, 0.01)
        plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mean.numpy(), var.numpy()),
                 label=r'$\mu_' + str(param_number) + r'=' + str(round(mean.item(), 2)) +
                       r',\ \sigma_' + str(param_number) + '=' + str(round(var.item(), 2)) + r'$')
        param_number += 1
    if show:
        plt.legend()
        plt.show()


#  N observations, K parameters = 2
def generate_data(n_observations: int, k_parameters=2, show=False):
    gaussian_distributions = []
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(MU_START, MU_END))
        sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
        normal_dist = torch.distributions.Normal(mu, sigma)
        sample = normal_dist.sample((n_observations, 1)).squeeze()
        gaussian_distributions.append(sample)
    for distribution in gaussian_distributions:
        plot_observation(distribution)
    if show:
        plt.legend()
        plt.show()
    return gaussian_distributions


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
def calculate_membership_dataset(x_dataset, parameters_matrix, one_hot=True):
    likelihood_matrix = []
    for dataset in x_dataset:
        for data in dataset:
            data_likelihood = []
            for matrix in parameters_matrix:
                mu = matrix[0]
                sigma = matrix[1]
                likelihood = calculate_likelihood_gaussian_observation(data.item(), mu.item(), sigma.item())
                data_likelihood.append(likelihood)
            if one_hot:
                for index in range(len(data_likelihood)):
                    data_likelihood[index] = 0 if data_likelihood[index] != max(data_likelihood) else 1
            likelihood_matrix.append(data_likelihood)
    likelihood_matrix = torch.tensor(likelihood_matrix)
    return likelihood_matrix


#  calculate_membership_dataset(x_dataset, parameters_matrix) -> membership_data
def recalculate_parameters(x_dataset, membership_data):
    membership_data = torch.transpose(membership_data, 0, 1)
    complete_dataset = torch.Tensor()
    new_parameters = []
    for dataset in x_dataset:
        complete_dataset = torch.cat((complete_dataset, dataset))
    for k in membership_data:
        data_set_one = []
        for one_hot_data in range(len(k)):
            if k[one_hot_data].item() == 1:
                data_set_one.append(complete_dataset[one_hot_data])
        data_set_one = torch.Tensor(data_set_one)
        mu = torch.mean(data_set_one)
        sigma = math.sqrt(torch.mean(data_set_one))
        #  In case no data in the dataset matched the distribution, re-initialize random parameters.
        if mu.item() != mu.item() or sigma != sigma:
            params = init_random_parameters(1)
            mu = params[0][0]
            sigma = params[0][1]
            new_parameters.append([mu.item(), sigma.item()])
        else:
            new_parameters.append([mu.item(), sigma])
    new_parameters = torch.Tensor(new_parameters)
    return new_parameters


def expectation_maximization(observations=200, k_parameters=2, iterations=5):
    my_data = generate_data(observations, k_parameters, show=True)
    parameters = init_random_parameters(k_parameters)
    print("Initial parameters: " + str(parameters))
    plot_gaussian_distribution_and_observations(parameters, my_data, show=True)
    for iteration in range(iterations):
        print("Iteration #" + str(iteration))
        membership_data = calculate_membership_dataset(my_data, parameters, one_hot=True)
        print("Membership dataset: " + str(membership_data))
        parameters = recalculate_parameters(my_data, membership_data)
        print("New parameters: " + str(parameters))
        plot_gaussian_distribution_and_observations(parameters, my_data, show=True)


expectation_maximization()
