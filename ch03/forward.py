import numpy as np
from activation import sigmoid

# Forward pass example


def init():
    nn = {}
    nn["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    nn["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    nn["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])

    nn["b1"] = np.array([0.1, 0.2, 0.3])
    nn["b2"] = np.array([0.1, 0.2])
    nn["b3"] = np.array([0.1, 0.2])

    return nn


def forward(nn, x):
    W1, W2, W3 = nn["W1"], nn["W2"], nn["W3"]
    b1, b2, b3 = nn["b1"], nn["b2"], nn["b3"]

    z1 = sigmoid(np.dot(x, W1) + b1)
    z2 = sigmoid(np.dot(z1, W2) + b2)
    z3 = np.dot(z2, W3) + b3
    return z3


nn = init()
x = np.array([1.0, 0.5])
y = forward(nn, x)
print(y)
