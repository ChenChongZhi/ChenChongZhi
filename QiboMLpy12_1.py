# Let's do the imports first

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn

from qibo import Circuit, gates, construct_backend

from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR

# Prepare the training dataset
def f(x):
    return 1 * torch.sin(x)  ** 2 - 0.3 * torch.cos(x)

num_samples = 30
x_train = torch.linspace(0, 2 * np.pi, num_samples, dtype=torch.float64).unsqueeze(1)
y_train = f(x_train)

# Normalizing it to be between -1 and 1 (because we are going to use a Z observable in our decoding)
y_train = 2 * ( (y_train - min(y_train)) / (max(y_train) - min(y_train)) - 0.5 )

# A plotting function which will be useful now and later
def plot_target(x, target, predictions=None):
    """Plot target function and, optionally, the predictions of our model."""
    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    plt.plot(
        x_train,
        y_train,
        marker=".",
        markersize=10,
        color="blue",
        label="Targets",
        alpha=0.7
    )
    if predictions is not None:
        plt.plot(
            x_train,
            y_pred.detach().numpy(),
            marker=".",
            markersize=10,
            color="red",
            label="Predictions",
            alpha=0.7
        )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.show()

# Plotting
plot_target(x=x_train, target=y_train)
