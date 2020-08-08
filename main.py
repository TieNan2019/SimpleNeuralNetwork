#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# __author__ = "HX"
# __date__ = "2020-03-21"
# __modified__ = "2020-03-21"
import numpy as np
import matplotlib.pyplot as plt


def dataset_init():
    """
        示例数据准备
    """
    sigma = 0.05
    # A 的结果应该为 0
    A = np.zeros([100, 2])
    A[0:50, :] = sigma * np.random.randn(50, 2) + np.array([-1, 1])
    A[50:100, :] = sigma * np.random.randn(50, 2) + np.array([1, -1])

    # B 的结果应该为 1
    B = np.zeros([100, 2])
    B[0:50, :] = sigma * np.random.randn(50, 2) + np.array([1, 1])
    B[50:100, :] = sigma * np.random.randn(50, 2) + np.array([-1, -1])

    return A, B


class NeuralNetwork(object):
    """
        反向传播神经网络工具类。
        损失函数采用均方误差计算，暂时不可定制。
        参数更新采用梯度下降法，日后可能会根据需
        要加入随机梯度下降和 Mini Batch
        学习率暂时为常数。
        :param X: 矩阵，用于训练的数据，每一行都为一组特征
        :param Y: 矩阵，用于训练的数据的标签，与训练数据相对应
        :param layers: 列表，同时包含神经网络层数和神经元个数的信息，
                       每个元素都为整数，代表该层的神经元个数
        :param activations: 列表，激活函数表，每个元素都为字符串，代
                             表该层神经元的激活函数
        :param alpha: 浮点数，学习率，每次更新参数的步长
        :param epoch: 整数，学习周期，最大更新参数的次数
    """
    def __init__(self, X, Y, layers, activations,
            alpha=0.05, epoch=200000):
        # 从数据中解析参数
        self.num_of_data, self.dimensions = X.shape
        self.depth = len(layers)
        self.width = max(layers)

        self.layers = layers

        self.X = X
        self.Y = Y
        
        # 初始化权重偏置
        self.weights = []
        self.bais = []
        dim = [self.dimensions] + self.layers
        for i in range(self.depth):
            self.weights.append(np.random.randn(dim[i+1], dim[i]))
            self.bais.append(np.random.randn(dim[i+1], 1))

        # 初始化激活函数
        self.activations = [None] * self.depth
        dict_for_activations = {
                "sigmoid": self.sigmoid,
                "ReLU": self.ReLU
                }
        for i in range(self.depth):
            self.activations[i] = dict_for_activations[activations[i]]

        # 用于存放临时计算结果
        self.z = [None] * self.depth
        self.a = [None] * self.depth

        # 初始化训练参数
        self.alpha = alpha
        self.epoch = int(epoch)

        # 反向传播临时计算结果
        self.delta = [None] * self.depth

        # 初始化激活函数导数
        self.activation_primes = [None] * self.depth
        dict_for_activations_prime = {
                "sigmoid": self.sigmoid_prime,
                "ReLU": self.ReLU_prime
                }
        for i in range(self.depth):
            self.activation_primes[i] = dict_for_activations_prime[activations[i]]

    def classify(self, x):
        """
            通过现有的参数，对样本的种类进行预测
            :param x: 矩阵，每一行都是一个样本的特征
            :return: 矩阵，由数据得到的预测结果
        """
        for i in range(self.depth):
            if i == 0:
                self.z[0] = self.weights[0] * x.T + self.bais[0]
            else:
                self.z[i] = self.weights[i] * self.a[i-1] + self.bais[i]
            self.a[i] = self.activations[i](self.z[i])
            
        return self.a[-1]

    def back_propagation(self, E):
        """
            反向传播算法。
            依次计算误差对每一个未激活的函数的偏导
            :param E: 矩阵，每一行都为预测结果和标记结果的差
        """
        self.delta[-1] = np.multiply(E, self.activation_primes[-1](self.z[-1]).T)
        for i in range(self.depth-2, -1, -1):
            self.delta[i] = np.multiply(self.delta[i+1] * self.weights[i+1], 
                    self.activation_primes[i](self.z[i]).T)

    def update(self):
        """
            参数更新，根据梯度下降算法得出的结果，对权重值和
            偏置进行修正
        """
        y = self.classify(self.X).T
        E = y - self.Y
        loss = np.linalg.norm(E)**2 / (2 * self.num_of_data)

        # 反向传播
        self.back_propagation(E)
        
        # 参数更新
        # 更新偏置
        for i in range(self.depth):
            self.bais[i] -= self.alpha * np.mean(self.delta[i].T, axis=1)

        # 更新权重
        self.weights[0] -= self.alpha * (self.delta[0].T * self.X) / self.num_of_data
        for i in range(1, self.depth):
            self.weights[i] -= self.alpha * (self.delta[i].T * self.a[i-1].T) / self.num_of_data

        return loss

    def fit(self):
        """
            通过已经载入的训练数据开始拟合
        """
        for i in range(self.epoch):
            loss = self.update()
            if i % 1000 == 0:
                print("Epoch %d : loss : %f" %(i, loss))
        
    def sigmoid(self, x):
        """
            sigmoid 激活函数
        """
        return 1 / (np.exp(-x) + 1)

    def ReLU(self, x):
        """
            线性整流激活函数
        """
        return np.multiply((x > 0), x)

    def sigmoid_prime(self, x):
        """
            sigmoid 函数的导数
        """
        return np.multiply(self.sigmoid(x), (1 - self.sigmoid(x)))

    def ReLU_prime(self, x):
        """
            线性整流函数的导数
        """
        return np.int32(x > 0)


if __name__ == "__main__":
    """
        主程序入口
    """
    A, B = dataset_init()
    X = np.mat(np.zeros([200, 2]))
    X[0:100, :] = A
    X[100:200, :] = B
    # A 相同结果为 0
    Y = np.mat(np.zeros([200, 1]))
    # B 不同结果为 1
    Y[0:100, :] = np.mat(np.ones([100, 1]))

    classifier = NeuralNetwork(X, Y, [2, 1], ["ReLU", "sigmoid"], alpha=0.2)
    classifier.fit()

