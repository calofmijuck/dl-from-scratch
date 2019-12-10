import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Relu(x):
    return np.maximum(0, x)


input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

x = input_data

func = sigmoid
# func = Relu
# func = np.tanh

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)  # Xavier Initialization
    # w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)  # He Initialization

    a = np.dot(x, w)
    z = func(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.savefig("fig.png")

