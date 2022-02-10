#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :logistic_regression.py
@Author  :JackHCC
@Date    :2022/2/5 19:59 
@Desc    :Implement Logistic Regression and Maximum Entropy Model

'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from scipy.optimize import fminbound
import copy
from collections import defaultdict
from scipy.misc import derivative
from math import e, log

# 图像显示中文
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']


class LogisticRegression:
    def __init__(self, max_iter=10000, distance=3, epsilon=1e-6):
        """
        Logistic回归
        :param max_iter: 最大迭代次数
        :param distance: 一维搜索的长度范围
        :param epsilon: 迭代停止阈值
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        # 权重
        self.w = None
        self.distance = distance
        self._X = None
        self._y = None

    @staticmethod
    def preprocessing(X):
        """将原始X末尾加上一列，该列数值全部为1"""
        row = X.shape[0]
        y = np.ones(row).reshape(row, 1)
        return np.hstack((X, y))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def grad(self, w):
        z = np.dot(self._X, w.T)
        grad = self._X * (self._y - self.sigmoid(z))
        grad = grad.sum(axis=0)
        return grad

    def likelihood_func(self, w):
        z = np.dot(self._X, w.T)
        f = self._y * z - np.log(1 + np.exp(z))
        return np.sum(f)

    def fit(self, data_x, data_y):
        self._X = self.preprocessing(data_x)
        self._y = data_y.T
        # （1）取初始化w
        w = np.array([[0] * self._X.shape[1]], dtype=np.float64)
        k = 0
        # （2）计算f(w)
        fw = self.likelihood_func(w)
        for _ in range(self.max_iter):
            # 计算梯度g(w)
            grad = self.grad(w)
            # （3）当梯度g(w)的模长小于精度时，停止迭代
            if (np.linalg.norm(grad, axis=0, keepdims=True) < self.epsilon).all():
                self.w = w
                break

            # 梯度方向的一维函数
            def f(x):
                z = w - np.dot(x, grad)
                return -self.likelihood_func(z)

            # （3）进行一维搜索，找到使得函数最大的lambda
            _lambda = fminbound(f, -self.distance, self.distance)

            # （4）设置w(k+1)
            w1 = w - np.dot(_lambda, grad)
            fw1 = self.likelihood_func(w1)

            # （4）当f(w(k+1))-f(w(k))的二范数小于精度，或w(k+1)-w(k)的二范数小于精度
            if np.linalg.norm(fw1 - fw) < self.epsilon or \
                    (np.linalg.norm((w1 - w), axis=0, keepdims=True) < self.epsilon).all():
                self.w = w1
                break

            # （5） 设置k=k+1
            k += 1
            w, fw = w1, fw1

        self.grad_ = grad
        self.n_iter_ = k
        self.coef_ = self.w[0][:-1]
        self.intercept_ = self.w[0][-1]

    def predict(self, x):
        p = self.sigmoid(np.dot(self.preprocessing(x), self.w.T))
        p[np.where(p > 0.5)] = 1
        p[np.where(p < 0.5)] = 0
        return p

    def score(self, X, y):
        y_c = self.predict(X)
        # 计算准确率
        error_rate = np.sum(np.abs(y_c - y.T)) / y_c.shape[0]
        return 1 - error_rate

    def draw(self, X, y):
        # 分隔正负实例点
        y = y[0]
        X_po = X[np.where(y == 1)]
        X_ne = X[np.where(y == 0)]
        # 绘制数据集散点图
        ax = plt.axes(projection='3d')
        x_1 = X_po[0, :]
        y_1 = X_po[1, :]
        z_1 = X_po[2, :]
        x_2 = X_ne[0, :]
        y_2 = X_ne[1, :]
        z_2 = X_ne[2, :]
        ax.scatter(x_1, y_1, z_1, c="r", label="正实例")
        ax.scatter(x_2, y_2, z_2, c="b", label="负实例")
        ax.legend(loc='best')
        # 绘制透明度为0.5的分隔超平面
        x = np.linspace(-3, 3, 3)
        y = np.linspace(-3, 3, 3)
        x_3, y_3 = np.meshgrid(x, y)
        a, b, c, d = self.w[0]
        z_3 = -(a * x_3 + b * y_3 + d) / c
        # 调节透明度
        ax.plot_surface(x_3, y_3, z_3, alpha=0.5)
        plt.show()


class MaxEntDFP:
    def __init__(self, epsilon, max_iter=1000, distance=0.01):
        """
        最大熵的DFP算法
        :param epsilon: 迭代停止阈值
        :param max_iter: 最大迭代次数
        :param distance: 一维搜索的长度范围
        """
        self.distance = distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.w = None
        self._dataset_X = None
        self._dataset_y = None
        # 标签集合，相当去去重后的y
        self._y = set()
        # key为(x,y), value为对应的索引号ID
        self._xyID = {}
        # key为对应的索引号ID, value为(x,y)
        self._IDxy = {}
        # 经验分布p(x,y)
        self._pxy_dic = defaultdict(int)
        # 样本数
        self._N = 0
        # 特征键值(x,y)的个数
        self._n = 0
        # 实际迭代次数
        self.n_iter_ = 0

    # 初始化参数
    def init_params(self, X, y):
        self._dataset_X = copy.deepcopy(X)
        self._dataset_y = copy.deepcopy(y)
        self._N = X.shape[0]

        for i in range(self._N):
            xi, yi = X[i], y[i]
            self._y.add(yi)
            for _x in xi:
                self._pxy_dic[(_x, yi)] += 1

        self._n = len(self._pxy_dic)
        # 初始化权重w
        self.w = np.zeros(self._n)

        for i, xy in enumerate(self._pxy_dic):
            self._pxy_dic[xy] /= self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def calc_zw(self, X, w):
        """书中第100页公式6.23，计算Zw(x)"""
        zw = 0.0
        for y in self._y:
            zw += self.calc_ewf(X, y, w)
        return zw

    def calc_ewf(self, X, y, w):
        """书中第100页公式6.22，计算分子"""
        sum_wf = self.calc_wf(X, y, w)
        return np.exp(sum_wf)

    def calc_wf(self, X, y, w):
        sum_wf = 0.0
        for x in X:
            if (x, y) in self._pxy_dic:
                sum_wf += w[self._xyID[(x, y)]]
        return sum_wf

    def calc_pw_yx(self, X, y, w):
        """计算Pw(y|x)"""
        return self.calc_ewf(X, y, w) / self.calc_zw(X, w)

    def calc_f(self, w):
        """计算f(w)"""
        fw = 0.0
        for i in range(self._n):
            x, y = self._IDxy[i]
            for dataset_X in self._dataset_X:
                if x not in dataset_X:
                    continue
                fw += np.log(self.calc_zw(x, w)) - self._pxy_dic[(x, y)] * self.calc_wf(dataset_X, y, w)

        return fw

    # DFP算法
    def fit(self, X, y):
        self.init_params(X, y)

        def calc_dfw(i, w):
            """计算书中第107页的拟牛顿法f(w)的偏导"""

            def calc_ewp(i, w):
                """计算偏导左边的公式"""
                ep = 0.0
                x, y = self._IDxy[i]
                for dataset_X in self._dataset_X:
                    if x not in dataset_X:
                        continue
                    ep += self.calc_pw_yx(dataset_X, y, w) / self._N
                return ep

            def calc_ep(i):
                """计算关于经验分布P(x,y)的期望值"""
                (x, y) = self._IDxy[i]
                return self._pxy_dic[(x, y)]

            return calc_ewp(i, w) - calc_ep(i)

        # 算出g(w)，是n*1维矩阵
        def calc_gw(w):
            return np.array([[calc_dfw(i, w) for i in range(self._n)]]).T

        # （1）初始正定对称矩阵，单位矩阵
        Gk = np.array(np.eye(len(self.w), dtype=float))

        # （2）计算g(w0)
        w = self.w
        gk = calc_gw(w)
        # 判断gk的范数是否小于阈值
        if np.linalg.norm(gk, ord=2) < self.epsilon:
            self.w = w
            return

        k = 0
        for _ in range(self.max_iter):
            # （3）计算pk
            pk = -Gk.dot(gk)

            # 梯度方向的一维函数
            def _f(x):
                z = w + np.dot(x, pk).T[0]
                return self.calc_f(z)

            # （4）进行一维搜索，找到使得函数最小的lambda
            _lambda = fminbound(_f, -self.distance, self.distance)

            delta_k = _lambda * pk
            # （5）更新权重
            w += delta_k.T[0]

            # （6）计算gk+1
            gk1 = calc_gw(w)
            # 判断gk1的范数是否小于阈值
            if np.linalg.norm(gk1, ord=2) < self.epsilon:
                self.w = w
                break
            # 根据DFP算法的迭代公式（附录B.24公式）计算Gk
            yk = gk1 - gk
            Pk = delta_k.dot(delta_k.T) / (delta_k.T.dot(yk))
            Qk = Gk.dot(yk).dot(yk.T).dot(Gk) / (yk.T.dot(Gk).dot(yk)) * (-1)
            Gk = Gk + Pk + Qk
            gk = gk1

            # （7）置k=k+1
            k += 1

        self.w = w
        self.n_iter_ = k

    def predict(self, x):
        result = {}
        for y in self._y:
            prob = self.calc_pw_yx(x, y, self.w)
            result[y] = prob

        return result


def newton_method_linear(func, args=(), error=1e-6, dx=1e-6):
    """一维牛顿法求f(x)=0的值

    :param func: 目标函数
    :param args: 参数列表
    :param error: 容差
    :param dx: 计算导数时使用的dx
    :return:
    """
    x0, y0 = 0, func(0, *args)
    while True:
        d = derivative(func, x0, args=args, dx=dx)  # 计算一阶导数
        x1 = x0 - y0 / d
        if abs(x1 - x0) < error:
            return x1
        x0, y0 = x1, func(x1, *args)


def improved_iterative_scaling(x, y, features, error=1e-6):
    """改进的迭代尺度法求最大熵模型

    :param x: 输入变量
    :param y: 输出变量
    :param features: 特征函数列表
    :return:
    """
    n_samples = len(x)  # 样本数量
    n_features = len(features)  # 特征函数数量

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 不同的x的数量
    n_y = len(y_list)  # 不同的y的数量
    n_total = n_x * n_y  # 不同样本的总数

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    print("边缘分布的经验分布", d2)

    # 计算特征函数关于经验分布的期望值:EP(fi) (empirical_joint_distribution_each_feature)
    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    d3 = [0.0] * n_features  # empirical_joint_distribution_each_feature
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features
    for j in range(n_features):
        for xi in range(n_x):
            for yi in range(n_y):
                if features[j](list(x_list[xi]), y_list[yi]):
                    d3[j] += d1[xi][yi]
                    nn[xi][yi] += 1

    print("特征函数关于经验分布的期望值:", d3)
    print("所有特征在(x,y)出现的次数:", nn)

    # 定义w的初值和模型P(Y|X)的初值
    w0 = [0] * n_features  # w的初值：wi=0
    p0 = [[1 / n_total] * n_y for _ in range(n_x)]  # 当wi=0时，P(Y|X)的值

    change = True
    while change:
        change = False

        # 遍历各个特征条件以更新w
        for j in range(n_features):
            def func(d, jj):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        if features[j](list(x_list[xxi]), y_list[yyi]):
                            res += d2[xxi] * p0[xxi][yyi] * pow(e, d * nn[xxi][yyi])
                res -= d3[jj]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(j,))

            # 更新wi的值
            w0[j] += dj
            if abs(dj) >= error:
                change = True

        # 计算新的模型
        p1 = [[0.0] * n_y for _ in range(n_x)]
        for xi in range(n_x):
            for yi in range(n_y):
                for j in range(n_features):
                    if features[j](list(x_list[xi]), y_list[yi]):
                        p1[xi][yi] += w0[j]
                p1[xi][yi] = pow(e, p1[xi][yi])
            total = sum(p1[xi][yi] for yi in range(n_y))
            if total > 0:
                for yi in range(n_y):
                    p1[xi][yi] /= total

        if not change:
            ans = {}
            for xi in range(n_x):
                for yi in range(n_y):
                    ans[(tuple(x_list[xi]), y_list[yi])] = p1[xi][yi]
            return w0, ans

        p0 = p1


if __name__ == '__main__':
    # Test Logistic Regression
    print("开始测试逻辑斯蒂回归……")
    # 训练数据集
    X_train = np.array([[3, 3, 3], [4, 3, 2], [2, 1, 2], [1, 1, 1], [-1, 0, 1], [2, -2, 1]])
    y_train = np.array([[1, 1, 1, 0, 0, 0]])
    # 构建实例，进行训练
    clf = LogisticRegression(epsilon=1e-6)
    clf.fit(X_train, y_train)
    clf.draw(X_train, y_train)
    print("迭代次数：{}次".format(clf.n_iter_))
    print("梯度：{}".format(clf.grad_))
    print("权重：{}".format(clf.w[0]))
    print("模型准确率：{:.2%}".format(clf.score(X_train, y_train)))

    # Test Max Entropy
    print("------------------------------------------")
    print("开始测试最大熵……")
    # 训练数据集
    dataset = np.array([['no', 'sunny', 'hot', 'high', 'FALSE'],
                        ['no', 'sunny', 'hot', 'high', 'TRUE'],
                        ['yes', 'overcast', 'hot', 'high', 'FALSE'],
                        ['yes', 'rainy', 'mild', 'high', 'FALSE'],
                        ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
                        ['no', 'rainy', 'cool', 'normal', 'TRUE'],
                        ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
                        ['no', 'sunny', 'mild', 'high', 'FALSE'],
                        ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
                        ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
                        ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
                        ['yes', 'overcast', 'mild', 'high', 'TRUE'],
                        ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
                        ['no', 'rainy', 'mild', 'high', 'TRUE']])

    X_train = dataset[:, 1:]
    y_train = dataset[:, 0]

    mae = MaxEntDFP(epsilon=1e-4, max_iter=1000, distance=0.01)
    mae.fit(X_train, y_train)
    print("模型训练迭代次数：{}次".format(mae.n_iter_))
    print("模型权重：{}".format(mae.w))

    result = mae.predict(['overcast', 'mild', 'high', 'FALSE'])
    print("预测结果：", result)

    print("------------------------------------------")
    print("开始测试一维牛顿法求 f(x) = 0 的值……")


    def f(x, k):
        return (x - k) ** 3

    print(newton_method_linear(f, args=(2,)))

    print("------------------------------------------")
    print("开始测试改进的迭代尺度算法 IIS……")
    dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]

    def f1(xx, yy):
        return xx == [1] and yy == 1

    def f2(xx, yy):
        return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)

    print(improved_iterative_scaling(dataset[0], dataset[1], [f1]))
    print(improved_iterative_scaling(dataset[0], dataset[1], [f2]))
