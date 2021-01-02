import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.optimizer import SGD, Adam
from common.multi_layer_net_extend import MultiLayerNet
from math import ceil

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True,
    )
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
    )
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = ceil(train_size / batch_size)
    epoch_cnt = 0

    for i in range(1, max_epochs * iter_per_epoch + 1):
    # for i in range(1, max_epochs * iter_per_epoch + 1):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for nt in (bn_network, network):
            grads = nt.gradient(x_batch, t_batch)
            optimizer.update(nt.params, grads)

        if i % iter_per_epoch == 0:
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc_list.append(bn_train_acc)
            train_acc_list.append(train_acc)

            print(
                "epoch: "
                + str(epoch_cnt)
                + " | "
                + str(train_acc)
                + " - "
                + str(bn_train_acc)
            )

            epoch_cnt += 1

    return train_acc_list, bn_train_acc_list


# Draw Graph
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print("====" + str(i + 1) + "/16" + "====")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, i + 1)
    plt.title("W:" + ("%.3f" % w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label="Batch Norm", markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", label="Normal", markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery = 2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    
plt.legend(loc='lower right')
plt.savefig("fig.png")
