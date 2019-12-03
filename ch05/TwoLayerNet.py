import sys, os

sys.path.append(os.pardir)
import numpy as np
from ch03.activation import sigmoid, softmax
from ch04.loss import cross_entropy
from ch04.gradient import numerical_gradient
from collections import OrderedDict
from layers import Affine, Relu, SoftmaxLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

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
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(p == t) / float(x.shape[0])

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["W2"] = self.layers["Affine2"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["b2"] = self.layers["Affine2"].db

        return grads
