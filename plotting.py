import math
import matplotlib.pyplot as plt
import numpy as np
import torch

x = torch.linspace(-20, 20, steps=256)  # x1
y = x ** 2
z = 2 ** x

fig, ax = plt.subplots()

ax.plot(x.numpy(), y.numpy(), linewidth=2.0)
ax.plot(x.numpy(), z.numpy(), linewidth=2.0)

ax.set(xlim=(-10, 10), xticks=np.arange(-10, 10),
       ylim=(-1, 8), yticks=np.arange(0, 20))

ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

plt.show()

