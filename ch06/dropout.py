import os, sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Decrease data size to artificially create overfitting
x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = False
# use_dropout = True
dropout_ratio = 0.2

# Create network
network = MultiLayerNet(
    input_size=784,
    hidden_size_list=[100, 100, 100, 100, 100, 100],
    output_size=10,
    use_dropout=use_dropout,
    dropout_ratio=dropout_ratio,
)

# Trainer
trainer = Trainer(
    network, x_train, t_train, x_test, t_test, epochs=201, mini_batch_size=100
)

trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker=markers["train"], label="train", markevery=10)
plt.plot(x, test_acc_list, marker=markers["test"], label="test", markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.savefig("figure.png")
plt.show()
