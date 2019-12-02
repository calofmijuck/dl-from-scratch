import sys, os

sys.path.append(os.pardir)

import numpy as np
from ch03.activation import sigmoid, softmax
from loss import cross_entropy
from gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        tmp = sigmoid(np.dot(x, W1) + b1)
        ret = softmax(np.dot(tmp, W2) + b2)

        return ret

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy(y, t)

    def numerical_gradient(self, x, t):
        f = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(f, self.params["W1"])
        grads["W2"] = numerical_gradient(f, self.params["W2"])
        grads["b1"] = numerical_gradient(f, self.params["b1"])
        grads["b2"] = numerical_gradient(f, self.params["b2"])

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        p = np.argmax(y, axis=1)
        ans = np.argmax(t, axis=1)

        acc = np.sum(p == ans) / float(x.shape[0])
        return acc

if __name__ == "__main__":
    net = TwoLayerNet(784, 100, 10)
    x = np.random.rand(100, 784)  # Dummy input data
    y = net.predict(x)
    print(y)
