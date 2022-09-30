import math
import torch
import random
from matplotlib import cm
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_color(i):
    colors = "rbgcmykw"
    return colors[i % len(colors)]


def init_random_parameters(k: int, expected_mean_std: torch.tensor):
    return generate_parameters(k, expected_mean_std, torch.zeros(k, 2))


def generate_parameters(k: int, rand_range: torch.tensor, base: torch.tensor):
    expected_mean = torch.zeros(k).uniform_(rand_range[0][0], rand_range[0][1])
    expected_std = torch.zeros(k).uniform_(rand_range[1][0], rand_range[1][1])
    generated_randoms = torch.t(torch.stack([expected_mean, expected_std]))

    init_masc = torch.where(base == 0, 1, 0)
    row_validation = torch.sum(init_masc, 1).unsqueeze(1)
    masc = torch.where(row_validation == 2, torch.tensor([1, 1]), torch.tensor([0, 0]))
    random_parameters = base + ((masc * generated_randoms))

    return random_parameters


class Random_Data_Generator:

    def __init__(self, n: int, k: int, expected_mean_std: torch.tensor):
        self.n = n
        self.k = k
        self.expected_mean_std = expected_mean_std

    def set_default_to_variables(self):
        self.n_per_tensor = self.n // self.k
        self.generated_output = torch.tensor([])
        self.parameters = init_random_parameters(self.k, self.expected_mean_std)

    def generate_data(self, plot_data: bool):
        self.set_default_to_variables()
        self.y = torch.zeros(self.n_per_tensor)
        output = torch.stack(self.generate_data_aux(plot_data))
        self.generated_output = output.view(*output.shape[:0], -1, *output.shape[self.k:])

        if plot_data:
            plt.show()

    def generate_data_aux(self, plot_data, curr_dist: int = 0):
        n_aux = self.n_per_tensor
        mu = self.parameters[curr_dist][0]
        std = self.parameters[curr_dist][1]

        normal = torch.distributions.Normal(mu, std)
        x = normal.sample((n_aux, 1)).squeeze()
        # x = torch.normal(mean=mu, std=std, size=(1, n_aux))[0]

        if plot_data:
            xcord = torch.linspace(torch.min(x) - std, torch.max(x) + std, 100)
            plt.plot(xcord, stats.norm.pdf(xcord, mu, std))
            plt.scatter(x, self.y, c=get_color(curr_dist))

        res_aux = [x]

        if curr_dist + 1 < self.k:
            new_k = curr_dist + 1
            new_data = self.generate_data_aux(plot_data, new_k)
            res_aux.extend(new_data)

        return res_aux


def calculate_likelihood_per_group_of_parameters(parameters: torch.tensor, samples: torch.tensor):
    bpart = (1 / math.sqrt(2 * math.pi * parameters[1] ** 2))
    fpart = math.e ** -(((samples - parameters[0]) ** 2) / (2 * parameters[1] ** 2))

    return bpart * fpart


def calculate_likelihood(parameters: torch.tensor, samples: torch.tensor):
    mean = parameters[:, 0][:, None]
    std = parameters[:, 1][:, None]

    bpart = (1 / torch.sqrt(2 * math.pi * std ** 2))
    fpart = math.e ** (-(1 / 2) * ((samples.repeat(2, 1) - mean) / std) ** 2)

    return torch.nan_to_num(bpart * fpart)


def calculate_membership_dataset(parameters: torch.tensor, samples: torch.tensor):
    original = calculate_likelihood(parameters, samples)
    transpose_o = torch.t(original)
    maxvalues = torch.amax(transpose_o, 1)
    return torch.where(original == maxvalues, 1.0, 0.0)


def recalculate_parameters(one_hot_vector, samples):
    values_per_membership = one_hot_vector * samples
    transpose = torch.t(values_per_membership)

    n_aux = torch.count_nonzero(transpose, 0)
    mean = torch.sum(values_per_membership, 1) / n_aux

    anti_neg_mean = torch.where(transpose == 0, 1, 0)
    anti_neg_mean = (anti_neg_mean * mean) + transpose

    std = torch.sqrt(torch.sum(torch.t((anti_neg_mean - mean) ** 2), 1) / n_aux)
    return torch.nan_to_num(torch.t(torch.stack([mean, std])))


expected_mean_std = torch.tensor([[10.0, 100.0], [3.1, 5.2]])
generator = Random_Data_Generator(200, 2, expected_mean_std)
generator.generate_data(False)
curr_parameters = init_random_parameters(generator.k, expected_mean_std)

print('Target parameters ', generator.parameters)

for i in range(5):
    print('Parameter on attempt #', i, ': ', curr_parameters)
    xcord = torch.linspace(0, 125, 100)
    plt.plot(xcord, stats.norm.pdf(xcord, curr_parameters[0][0], curr_parameters[0][1]))
    plt.plot(xcord, stats.norm.pdf(xcord, curr_parameters[1][0], curr_parameters[1][1]))
    plt.scatter(generator.generated_output, torch.zeros(generator.n), c='b')
    plt.show()

    membership = calculate_membership_dataset(curr_parameters, generator.generated_output)
    curr_parameters = recalculate_parameters(membership, generator.generated_output)
    curr_parameters = generate_parameters(generator.k, expected_mean_std, curr_parameters)
