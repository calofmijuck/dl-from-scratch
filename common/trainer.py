import sys, os

sys.path.append(os.pardir)

import numpy as np
from common.optimizer import *
from tqdm import tqdm


# Class that trains the network
class Trainer:
    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
        evaluate_sample_num_per_epoch=None,
        verbose=True,
    ):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_per_epoch = evaluate_sample_num_per_epoch

        # Optimizer
        optimizer_class_dict = {
            "sgd": SGD,
            "momentum": Momentum,
            "nesterov": Nesterov,
            "adagrad": AdaGrad,
            "rmsprop": RMSProp,
            "adam": Adam,
        }
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # Create training data
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # Feed forward, calculate gradient, backprop and update parameters
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

    def run_epoch(self, epoch):
        print("--- epoch {} ---".format(epoch))
        for _ in tqdm(range(self.iter_per_epoch)):
            self.train_step()

        x_train_sample, t_train_sample = self.x_train, self.t_train
        x_test_sample, t_test_sample = self.x_test, self.t_test

        # Number of samples to evaluate per epoch
        if not self.evaluate_sample_per_epoch is None:
            t = self.evaluate_sample_per_epoch
            x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
            x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

        # Calculate accuracy for train data and test data
        train_acc = self.network.accuracy(x_train_sample, t_train_sample)
        test_acc = self.network.accuracy(x_test_sample, t_test_sample)
        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)

        if self.verbose:
            print("train acc: {:.5f}, test acc: {:.5f}".format(train_acc, test_acc))

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch)

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("Final Test Accuracy: " + str(test_acc))
