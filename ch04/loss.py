import numpy as np


def mean_squared(x, t):
    return 0.5 * np.sum((x - t) ** 2)


# def cross_entropy(x, t):
#     ep = 1e-7
#     return -np.sum(t * np.log(x + ep))


def cross_entropy(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    if t.size == x.size:
        t = t.argmax(axis=1)

    ep = 1e-7
    batch_size = x.shape[0]
    return -np.sum(np.log(x[np.arange(batch_size), t] + ep)) / batch_size
