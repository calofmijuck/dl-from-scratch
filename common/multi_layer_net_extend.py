import sys, os

sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import (
    Affine,
    Relu,
    Sigmoid,
    BatchNormalization,
    SoftmaxLoss,
    Dropout,
)


class MultiLayerNet:
    """
    Parameters
    ------------------------
    use_dropout
    dropout_ratio
    use_batchnorm
    """

    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
        use_dropout=False,
        dropout_ratio=0.5,
        use_batchnorm=False,
    ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # Weight initialization
        self.__init_weight(weight_init_std)

        # Create layers
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )

            if self.use_batchnorm:
                self.params["gamma" + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params["beta" + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers["BatchNorm" + str(idx)] = BatchNormalization(
                    self.params["gamma" + str(idx)], self.params["beta" + str(idx)]
                )

            self.layers["Activation_function" + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_ratio)

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )

        self.last_layer = SoftmaxLoss()

    def __init_weight(self, weight_init_std):
        # Can be set to He or Xavier
        size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        scale = weight_init_std
        val = str(weight_init_std).lower()
        for idx in range(1, len(size_list)):
            if val in ("relu", "he"):  # Relu / He initialization
                scale = np.sqrt(2.0 / size_list[idx - 1])
            elif val in ("sigmoid", "xavier"):  # Sigmoid / Xavier initialization
                scale = np.sqrt(1.0 / size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(
                size_list[idx - 1], size_list[idx]
            )
            self.params["b" + str(idx)] = np.zeros(size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        # Forward
        self.loss(x, t, train_flg=True)

        # Backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = (
                self.layers["Affine" + str(idx)].dW
                + self.weight_decay_lambda * self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads["gamma" + str(idx)] = self.layers["BatchNorm" + str(idx)].dgamma
                grads["beta" + str(idx)] = self.layers["BatchNorm" + str(idx)].dbeta

        return grads
