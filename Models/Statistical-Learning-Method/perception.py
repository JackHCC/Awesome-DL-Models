#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Statistical-Learning-Method 
@File    :perception.py
@Author  :JackHCC
@Date    :2022/1/13 15:05 
@Desc    :Implement Perception Machine
'''
import numpy as np
from matplotlib import pyplot as plt
import logging


class Perception:
    def __init__(self, X, Y, lr=0.001, plot_enable=True):
        """
        Init perception.

        :param X: feature vector
        :param Y: label
        :param lr: learning rate
        :param plot_enable: plot or not
        """
        self.X = X
        self.Y = Y
        self.lr = lr
        self.plot_en = plot_enable
        if plot_enable:
            self.__model_plot = DisplayModel(self.X, self.Y)
            self.__model_plot.open_in()

    def param_init(self, method="random"):
        """
        Weight and bias init

        :param method: random or zero
        :return: weight and b
        """
        if method == "zero":
            weight = np.zeros(self.X.shape[1])
            bias = 0
        elif method == "random":
            weight = np.random.random(self.X.shape[1])
            bias = np.random.random(1)
        else:
            logging.error("method parameter can only be random or zero!")
            return 0, 0

        print(weight, bias)
        return weight, bias

    def fit(self, param_init_method="random"):
        """
        Train the model
        :param param_init_method: random or zero
        :return: parameters of model
        """
        weight, bias = self.param_init(method=param_init_method)

        epoch = 0   # Train epoch

        error_flag = True   # Classification error identification
        while error_flag:
            error_flag = False
            for item in range(self.X.shape[0]):
                if self.plot_en:
                    self.__model_plot.plot(weight, bias, epoch)

                loss = self.Y[item] * (weight @ self.X[item] + bias)    # loss function

                if loss <= 0:
                    # update weight and bias
                    weight += self.lr * self.Y[item] * self.X[item]
                    bias += self.lr * self.Y[item]
                    epoch += 1
                    print("Epoch {}, weight = {}, bias = {}, formula = {}".format(epoch, weight, bias,
                                                                                  self.formula(weight, bias)))
                    # Classification error identifies the wrong classification point in this cycle, which is set to true
                    error_flag = True
                    break
        if self.plot_en:
            self.__model_plot.close()

        return weight, bias

    @staticmethod
    def formula(weight, bias):
        text = 'x1 ' if weight[0] == 1 else '%d*x1 ' % weight[0]
        text += '+ x2 ' if weight[1] == 1 else (
            '+ %d*x2 ' % weight[1] if weight[1] > 0 else '- %d*x2 ' % -weight[1])
        text += '= 0' if bias == 0 else ('+ %d = 0' % bias if bias > 0 else '- %d = 0' % -bias)
        return text


class DisplayModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def open_in():
        """
        Open interaction mode to display dynamic interaction diagram
        """
        plt.ion()

    @staticmethod
    def close():
        """
        Close interaction mode to display dynamic interaction diagram
        """
        plt.ioff()
        plt.show()

    def plot(self, weight, bias, epoch):
        """
        :param weight: array
        :param bias: array
        :param epoch: integer
        """
        plt.cla()
        # Here we assume that X has two characteristics
        plt.xlim(0, np.max(self.X.T[0]) + 1)
        plt.ylim(0, np.max(self.X.T[1]) + 1)

        scatter = plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        plt.legend(*scatter.legend_elements())

        if True in list(weight == 0):
            plt.plot(0, 0)
        else:
            # Calculate the two intersections of the hyperplane and the coordinate axis
            x1 = -bias / weight[0]
            x2 = -bias / weight[1]
            plt.plot([x1, 0], [0, x2])
            text = Perception.formula(weight, bias)
            plt.text(0.3, x2 - 0.1, text)

        plt.title('Epoch: %d' % epoch)
        plt.pause(0.01)


if __name__ == "__main__":
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    model = Perception(X, Y, lr=1)
    weight, bias = model.fit(param_init_method="random")

