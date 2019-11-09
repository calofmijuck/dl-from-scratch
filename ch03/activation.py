import numpy as np

# Supports np.array
def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    c = np.max(x)  # Guard for overflow
    e = np.exp(x - c)
    return e / np.sum(e)
