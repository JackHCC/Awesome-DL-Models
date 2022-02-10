#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :markov_chain_monte_carlo.py
@Author  :JackHCC
@Date    :2022/2/10 12:27 
@Desc    :Implement Metropolis Hastings and Gibbs Sampling

'''
from abc import ABC
from abc import abstractmethod
from collections import Counter
import numpy as np
from bisect import bisect_left
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class BaseDistribution(ABC):
    """随机变量分布的抽象基类"""

    @abstractmethod
    def pdf(self, x):
        """计算概率密度函数"""
        pass

    def cdf(self, x):
        """计算分布函数"""
        raise ValueError("未定义分布函数")


class UniformDistribution(BaseDistribution):
    """均匀分布

    :param a: 左侧边界
    :param b: 右侧边界
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a < x < self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        elif self.a <= x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1


class GaussianDistribution(BaseDistribution):
    """高斯分布（正态分布）

    :param u: 均值
    :param s: 标准差
    """

    def __init__(self, u, s):
        self.u = u
        self.s = s

    def pdf(self, x):
        return pow(np.e, -1 * (pow(x - self.u, 2)) / 2 * pow(self.s, 2)) / (np.sqrt(2 * np.pi * pow(self.s, 2)))


def direct_sampling_method(distribution, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """直接抽样法抽取样本

    :param distribution: 定义分布函数的概率分布
    :param n_samples: 样本数
    :param a: 定义域左侧边界
    :param b: 定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    for _ in range(n_samples):
        y = np.random.rand()

        # 二分查找解方程：F(x) = y
        l, r = a, b
        while r - l > tol:
            m = (l + r) / 2
            if distribution.cdf(m) > y:
                r = m
            else:
                l = m

        samples.append((l + r) / 2)

    return samples


def accept_reject_sampling_method(d1, d2, c, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """接受-拒绝抽样法

    :param d1: 目标概率分布
    :param d2: 建议概率分布
    :param c: 参数c
    :param n_samples: 样本数
    :param a: 建议概率分布定义域左侧边界
    :param b: 建议概率分布定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    waiting = direct_sampling_method(d2, n_samples * 2, a=a, b=b, tol=tol, random_state=random_state)  # 直接抽样法得到建议分布的样本
    while len(samples) < n_samples:
        if not waiting:
            waiting = direct_sampling_method(d2, (n_samples - len(samples)) * 2, a, b)

        x = waiting.pop()
        u = np.random.rand()
        if u <= (d1.pdf(x) / (c * d2.pdf(x))):
            samples.append(x)

    return samples


def get_stationary_distribution(P, tol=1e-8, max_iter=1000):
    """迭代法求离散有限状态马尔可夫链的某个平稳分布

    根据平稳分布的定义求平稳分布。如果有无穷多个平稳分布，则返回其中任意一个。如果不存在平稳分布，则无法收敛。

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 平稳分布
    """
    n_components = len(P)

    # 初始状态分布：均匀分布
    pi0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pi1 = np.dot(P, pi0)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pi0 - pi1)) < tol:
            break

        pi0 = pi1

    return pi0


def is_reducible(P):
    """计算马尔可夫链是否可约

    :param P: 转移概率矩阵
    :return: 可约 = True ; 不可约 = False
    """
    n_components = len(P)

    # 遍历所有状态k，检查从状态k出发能否到达任意状态
    for k in range(n_components):
        visited = set()  # 当前已遍历过的状态

        find = False  # 当前是否已找到可到达任意位置的时刻
        stat0 = (False,) * k + (True,) + (False,) * (n_components - k - 1)  # 时刻0可达到的位置

        while stat0 not in visited:
            visited.add(stat0)
            stat1 = [False] * n_components

            for j in range(n_components):
                if stat0[j] is True:
                    for i in range(n_components):
                        if P[i][j] > 0:
                            stat1[i] = True

            # 如果已经到达之前已检查可到达任意状态的状态，则不再继续寻找
            for i in range(k):
                if stat1[i] is True:
                    find = True
                    break

            # 如果当前时刻可到达任意位置，则不再寻找
            if all(stat1) is True:
                find = True
                break

            stat0 = tuple(stat1)

        if not find:
            return True

    return False


def is_periodic(P):
    """计算马尔可夫链是否有周期性

    :param P: 转移概率矩阵
    :return: 有周期性 = True ; 无周期性 = False
    """
    n_components = len(P)

    # 0步转移概率矩阵
    P0 = P.copy()
    hash_P = tuple(P0.flat)

    # 每一个状态上一次返回状态的时刻的最大公因数
    gcd = [0] * n_components

    visited = Counter()  # 已遍历过的t步转移概率矩阵
    t = 1  # 当前时刻t

    # 不断遍历时刻t，直至满足如下条件：当前t步转移矩阵之前已出现过2次（至少2次完整的循环）
    while visited[hash_P] < 2:
        visited[hash_P] += 1

        # 记录当前返回状态的状态
        for i in range(n_components):
            if P0[i][i] > 0:
                if gcd[i] == 0:  # 状态i出发时，还从未返回过状态i
                    gcd[i] = t
                else:  # 计算最大公约数
                    gcd[i] = np.gcd(gcd[i], t)

        # 检查当前时刻是否还有未返回(gcd[i]=0)或返回状态的所有时间长的最大公因数大于1(gcd[i]>1)的状态
        for i in range(n_components):
            if gcd[i] == 0 or gcd[i] > 1:
                break
        else:
            return False

        # 计算(t+1)步转移概率矩阵
        P1 = np.dot(P0, P)

        P0 = P1
        hash_P = tuple(P0.flat)
        t += 1

    return True


def get_stationary_distribution_ergodic(P, start_iter=1000, end_iter=2000, random_state=0):
    """遍历定理求离散有限状态马尔可夫链的某个平稳分布

    要求离散状态、有限状态马尔可夫链是不可约、非周期的。

    :param P: 转移概率矩阵
    :param start_iter: 认为多少次迭代之后状态分布就是平稳分布
    :param end_iter: 计算从start_iter次迭代到end_iter次迭代的状态分布
    :param random_state: 随机种子
    :return: 平稳分布
    """
    n_components = len(P)
    np.random.seed(random_state)

    # 计算累计概率用于随机抽样
    Q = P.T
    for i in range(n_components):
        for j in range(1, n_components):
            Q[i][j] += Q[i][j - 1]

    # 设置初始状态
    x = 0

    # start_iter次迭代
    for _ in range(start_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)

    F = np.zeros(n_components)
    # start_iter次迭代到end_iter次迭代
    for _ in range(start_iter, end_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)
        F[x] += 1

    return F / sum(F)


def is_reversible(P, tol=1e-4, max_iter=1000):
    """计算有限状态马尔可夫链是否可逆

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 可逆 = True ; 不可逆 = False
    """
    n_components = len(P)
    D = get_stationary_distribution(P, pow(tol, 2), max_iter)  # 计算平稳分布（来自迭代法求离散有限状态马尔可夫链的某个平稳分布）
    for i in range(n_components):
        for j in range(n_components):
            if not - tol < P[i][j] * D[j] - P[j][i] * D[i] < tol:
                return False
    return True


def metropolis_hastings_method(d1, func, m, n, x0, random_state=0):
    """Metroplis-Hastings算法抽取样本

    :param d1: 目标概率分布的概率密度函数
    :param func: 目标求均值函数
    :param n_features: 随机变量维度
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
    """
    np.random.seed(random_state)

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    n_features = len(x0)

    # 循环执行n次迭代
    for k in range(n):
        # 按照建议分布q(x,x')随机抽取一个候选状态
        # q(x,x')为均值为x，方差为1的正态分布
        x1 = np.random.multivariate_normal(x0, np.diag([1] * n_features), 1)[0]

        # 计算接受概率
        a = min(1, d1(x1) / d1(x0))

        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = np.random.rand()

        # 若u<=a，则转移状态；否则不转移
        if u <= a:
            x0 = x1

        # 收集样本集合
        if k >= m:
            samples.append(x0)
            sum_ += func(x0)

    return samples, sum_ / (n - m)


def single_component_metropolis_hastings_method(d1, func, m, n, x0, random_state=0):
    """单分量Metroplis-Hastings算法抽取样本

    :param d1: 目标概率分布的概率密度函数
    :param func: 目标求均值函数
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
    """
    np.random.seed(random_state)

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    n_features = len(x0)
    j = 0  # 当前正在更新的分量

    # 循环执行n次迭代
    for k in range(n):
        # 按照建议分布q(x,x')随机抽取一个候选状态
        # q(x,x')为均值为x，方差为1的正态分布
        x1 = x0.copy()
        x1[j] = np.random.multivariate_normal([x0[j]], np.diag([1]), 1)[0][0]

        # 计算接受概率
        a = min(1, d1(x1) / d1(x0))

        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = np.random.rand()

        # 若u<=a，则转移状态；否则不转移
        if u <= a:
            x0 = x1

        # 收集样本集合
        if k >= m:
            samples.append(x0)
            sum_ += func(x0)

        j = (j + 1) % n_features

    return samples, sum_ / (n - m)


def gibbs_sampling_method(mean, cov, func, n_samples, m=1000, random_state=0):
    """吉布斯抽样算法在二元正态分布中抽取样本

    与np.random.multivariate_normal方法类似

    :param mean: n元正态分布的均值
    :param cov: n元正态分布的协方差矩阵
    :param func: 目标求均值函数
    :param n_samples: 样本量
    :param m: 收敛步数
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    n_features = len(mean)

    # 选取初始样本
    x0 = mean

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    # 循环执行n次迭代
    for k in range(m + n_samples):
        # 根据满条件分布逐个抽取样本
        x0[0] = np.random.multivariate_normal([x0[1] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1)[0][0]
        x0[1] = np.random.multivariate_normal([x0[0] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1)[0][0]

        # 收集样本集合
        if k >= m:
            samples.append(x0.copy())
            sum_ += func(x0)

    return samples, sum_ / n_samples


def d1_pdf(x):
    """随机变量x=(x_1,x_2)的联合概率密度"""
    return x[0] * pow(np.e, -x[1]) if 0 < x[0] < x[1] else 0


def f(x):
    """目标求均值函数"""
    return x[0] + x[1]


if __name__ == "__main__":
    print("开始测试直接采样法……")
    distribution = UniformDistribution(-3, 3)
    samples = direct_sampling_method(distribution, 10, -3, 3)
    print([round(v, 2) for v in samples])  # [0.29, 1.29, 0.62, 0.27, -0.46, 0.88, -0.37, 2.35, 2.78, -0.7]

    print("------------------------------------------")
    print("开始测试接受-拒绝采样法……")
    d1 = GaussianDistribution(0, 1)
    d2 = UniformDistribution(-3, 3)

    c = (1 / np.sqrt(2 * np.pi)) / (1 / 6)  # 计算c的最小值
    samples = accept_reject_sampling_method(d1, d2, c, 10, a=-3, b=3)
    print([round(v, 2) for v in samples])  # [0.17, -0.7, -0.37, 0.88, -0.46, 0.27, 0.62, 0.29, 0.27, 0.62]

    print("------------------------------------------")
    print("开始测试马尔可夫链的某个平稳分布……")
    np.set_printoptions(precision=2, suppress=True)
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(get_stationary_distribution(P))  # [0.4 0.2 0.4]

    print("------------------------------------------")
    print("开始测试马尔可夫链是否可约……")
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(is_reducible(P))  # False
    P = np.array([[0, 0.5, 0],
                  [1, 0, 0],
                  [0, 0.5, 1]])
    print(is_reducible(P))  # True

    print("------------------------------------------")
    print("开始测试马尔可夫链是否有周期性……")
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(is_periodic(P))  # False
    P = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])
    print(is_periodic(P))  # True

    print("------------------------------------------")
    print("开始测试遍历定理求离散有限状态马尔可夫链的某个平稳分布……")
    np.set_printoptions(precision=2, suppress=True)
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(get_stationary_distribution_ergodic(P))  # [0.39 0.18 0.43]

    print("------------------------------------------")
    print("开始测试马尔可夫链是否可逆……")
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(is_reversible(P))  # True
    P = np.array([[0.25, 0.5, 0.25],
                  [0.25, 0, 0.5],
                  [0.5, 0.5, 0.25]])
    print(is_reversible(P))  # False

    print("------------------------------------------")
    print("开始测试Metropolis-Hastrings算法……")

    samples, avg = metropolis_hastings_method(d1_pdf, f, m=1000, n=11000, x0=[5, 8])

    # print(samples)  # [array([0.39102823, 0.58105655]), array([0.39102823, 0.58105655]), ...]
    print("样本目标函数均值:", avg)  # 4.720997790412456


    def draw_distribution():
        """绘制总体概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i][j] = d1_pdf([i / 10, j / 10])

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()


    def draw_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            if i < 10 and j < 10:
                Z[int(i // 0.1)][int(j // 0.1)] += 1

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()

    draw_sample()

    print("------------------------------------------")
    print("开始测试单分量Metropolis-Hastrings算法……")
    samples, avg = single_component_metropolis_hastings_method(d1_pdf, f, m=1000, n=11000, x0=[5, 8])

    # print(samples)  # [[0.6497854877644121, 1.597507333170185], [0.6497854877644121, 1.597507333170185], ...]
    print("样本目标函数均值:", avg)  # 4.7348085753536076


    def draw_single_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            if i < 10 and j < 10:
                Z[int(i // 0.1)][int(j // 0.1)] += 1

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()


    draw_single_sample()

    print("------------------------------------------")
    print("开始测试吉布斯抽样算法……")
    samples, avg = gibbs_sampling_method([0, 0], [[1, 0.5], [0.5, 1]], f, n_samples=10000)

    # print(samples)  # [[-2.0422584903207794, -2.5037388977869997], [-1.211915315832784, -1.4359343041313015], ...]
    print("样本目标函数均值:", avg)  # 0.0016714992469064399


    def draw_gibbs_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            Z[int(i // 0.1) + 50][int(j // 0.1) + 50] += 1

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()

    draw_gibbs_sample()
