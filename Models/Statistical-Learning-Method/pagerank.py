#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :pagerank.py
@Author  :JackHCC
@Date    :2022/2/10 16:02 
@Desc    :Implement PageRank

'''
import numpy as np


def pagerank_basic(M, tol=1e-8, max_iter=1000):
    """使用PageRank的基本定义求解PageRank值

    要求有向图是强联通且非周期性的

    :param M: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 初始状态分布：均匀分布
    pr0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pr1 = np.dot(M, pr0)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pr0 - pr1)) < tol:
            break

        pr0 = pr1

    return pr0


def pagerank_1(M, d=0.8, tol=1e-8, max_iter=1000):
    """PageRank的迭代算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 初始状态分布：均匀分布
    pr0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pr1 = d * np.dot(M, pr0) + (1 - d) / n_components

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pr0 - pr1)) < tol:
            break

        pr0 = pr1

    return pr0


def pagerank_2(M, d=0.8, tol=1e-8, max_iter=1000):
    """计算一般PageRank的幂法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 选择初始向量x0：均匀分布
    x0 = np.array([1 / n_components] * n_components)

    # 计算有向图的一般转移矩阵A
    A = d * M + (1 - d) / n_components

    # 迭代并规范化结果向量
    for _ in range(max_iter):
        x1 = np.dot(A, x0)
        x1 /= np.max(x1)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(x0 - x1)) < tol:
            break

        x0 = x1

    # 对结果进行规范化处理，使其表示概率分布
    x0 /= np.sum(x0)

    return x0


def pagerank_3(M, d=0.8, tol=1e-8, max_iter=1000):
    """PageRank的代数算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 计算第一项：(I-dM)^-1
    r1 = np.linalg.inv(np.diag([1] * n_components) - d * M)

    # 计算第二项：(1-d)/n 1
    r2 = np.array([(1 - d) / n_components] * n_components)

    return np.dot(r1, r2)


if __name__ == "__main__":
    print("开始测试PageRank基本定义求PageRank值……")
    np.set_printoptions(precision=2, suppress=True)
    P = np.array([[0, 1 / 2, 1, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])
    print(pagerank_basic(P))  # [0.33 0.22 0.22 0.22]

    print("------------------------------------------")
    print("开始测试PageRank的迭代算法……")
    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])
    print(pagerank_1(P))  # [0.1  0.13 0.13 0.13]

    print("------------------------------------------")
    print("开始测试计算一般PageRank的幂法……")
    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])
    print(pagerank_2(P))  # [0.2  0.27 0.27 0.27]

    print("------------------------------------------")
    print("开始测试PageRank的代数算法……")
    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])
    print(pagerank_3(P))  # [0.1  0.13 0.13 0.13]
