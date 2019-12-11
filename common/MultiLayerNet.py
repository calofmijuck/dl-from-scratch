import sys, os

sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from layers import *


class MultiLayerNet:
    """
    Parameters
    ------------------------
    input_size
    hidden_size_list : list of sizes of hidden layers
    output_size
    activation : activation function
    weight_init_std
    weight_decay_lambda
    """

    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
    ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # Weight initialization
        self.__init_weight(weight_init_std)

        # Create layers
        activation_layer = {"sigmoid": Sigmoid(), "relu": Relu()}
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            self.layers["Activation_function" + str(idx)] = activation_layer[activation]

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
            elif val in ("sigmoid", "xavier",):  # Sigmoid / Xavier initialization
                scale = np.sqrt(1.0 / size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(
                size_list[idx - 1], size_list[idx]
            )
            self.params["b" + str(idx)] = np.zeros(size_list[idx])

