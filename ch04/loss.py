import numpy as np


def mean_squared(x, t):
    return 0.5 * np.sum((x - t) ** 2)


def cross_entropy(x, t):
    ep = 1e-7
    return -np.sum(t * np.log(x + ep))

