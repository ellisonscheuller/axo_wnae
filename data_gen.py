import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random

# Fix seeds for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Number of bins and limits for plotting
n_bins_x = 20
n_bins_y = 30
x_min = -2
x_max = 2
y_min = -3
y_max = 3
x_array = np.linspace(x_min, x_max, n_bins_x+1)
y_array = np.linspace(y_min, y_max, n_bins_y+1)

samples, labels = make_s_curve(n_samples=1000, noise=0.1)
training_data, validation_data = train_test_split(
    samples[:, [0, 2]],
    test_size=0.2,
    shuffle=True,
)

# Plot the training data
plt.figure()
plt.scatter(training_data[:, 0], training_data[:, 1])
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
