import numpy as np
import matplotlib.pyplot as plt
from simple_conv_net import SimpleConvNet


def create_figure(idx, filters, x_size=6):
    FN = filters.shape[0]
    y_size = int(np.ceil(FN / x_size))

    fig = plt.figure(idx)
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(y_size, x_size, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


def show_weights(random, trained):
    create_figure(1, random)
    create_figure(2, trained)

    plt.show()


network = SimpleConvNet()
random_weight = np.copy(network.params['W1'])

network.load_params("params.pkl")
trained_weight = np.copy(network.params['W1'])

show_weights(random_weight, trained_weight)
