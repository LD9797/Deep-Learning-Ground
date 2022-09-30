import math
import matplotlib.pyplot as plt
import numpy as np
import torch


def t(n):
    if n == 2:
        return 2
    if n == 3:
        return 1
    if n > 3:
        return 2 * t(math.ceil((n+1)/2)) + t(math.ceil(n/2))


pairs = []
pairs_plus_one = []

for x in range(2, 100):
    pairs.append([x, t(x)])
    pairs_plus_one.append([x, t(x+1)])


pairs = np.array(pairs)
pairs_plus_one = np.array(pairs_plus_one)

plt.scatter(pairs[:, 0], pairs[:, 1], s=5, alpha=0.5)
plt.scatter(pairs_plus_one[:, 0], pairs_plus_one[:, 1], s=5, alpha=0.5)
plt.show()



