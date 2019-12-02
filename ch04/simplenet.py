import sys, os

sys.path.append(os.pardir)
import numpy as np
from gradient import numerical_gradient
from loss import cross_entropy
from ch03.activation import softmax


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y, t)

        return loss

if __name__ == "__main__":
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])
    net.loss(x, t)

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)
