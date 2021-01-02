import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Decrease training data size for fast results
x_train = x_train[:500]
t_train = t_train[:500]

# Separate validation data
validation_rate = 0.2
validation = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)

x_val = x_train[:validation]
t_val = t_train[:validation]
x_train = x_train[validation:]
t_train = t_train[validation:]

# Trainer method - need this to repeat the training process
def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10,
        weight_decay_lambda=weight_decay,
    )

    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epochs,
        mini_batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": lr},
        verbose=False,
    )

    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# Hyperparameter random search
trial = 100
results_val = {}
results_train = {}

for _ in range(trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    # Validation accuracy and train data accuracy
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    key = "lr: {:.10e}, weight decay: {:.10e}".format(lr, weight_decay)
    print("val acc: {:.2f} | ".format(val_acc_list[-1]) + key)

    # Save
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# Plot
graph = 20  # number of graphs
col = 5
row = int(np.ceil(graph / col))
i = 0

# Sort by the final validation accuracy, in descending order
s = sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True)

for key, val_acc_list in s:
    print("Top {:2d} (val acc: {:.2f}) | ".format(i + 1, val_acc_list[-1]) + key)

    plt.subplot(row, col, i + 1)
    plt.title("Top " + str(i + 1))
    plt.ylim(0, 1)
    if i % 5:
        plt.yticks([])
    plt.xticks([])

    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph:
        break

plt.show()
