#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :conditional_random_field.py
@Author  :JackHCC
@Date    :2022/2/10 14:32 
@Desc    :Implement CRF

'''
import math
import bisect
import random
from logistic_regression import newton_method_linear
import numpy as np
from scipy.misc import derivative


def golden_section_for_line_search(func, a0, b0, epsilon):
    """一维搜索极小值点（黄金分割法）
    :param func: [function] 一元函数
    :param a0: [int/float] 目标区域左侧边界
    :param b0: [int/float] 目标区域右侧边界
    :param epsilon: [int/float] 精度
    """
    a1, b1 = a0 + 0.382 * (b0 - a0), b0 - 0.382 * (b0 - a0)
    fa, fb = func(a1), func(b1)

    while b1 - a1 > epsilon:
        if fa <= fb:
            b0, b1, fb = b1, a1, fa
            a1 = a0 + 0.382 * (b0 - a0)
            fa = func(a1)
        else:
            a0, a1, fa = a1, b1, fb
            b1 = b0 - 0.382 * (b0 - a0)
            fb = func(b1)

    return (a1 + b1) / 2


def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点各个自变量的梯度向量（偏导数列表）
    :param func: [function] n元函数
    :param arr: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 偏导数
    """
    n_features = len(arr)
    ans = []
    for i in range(n_features):
        def f(x):
            arr2 = list(arr)
            arr2[i] = x
            return func(arr2)

        ans.append(derivative(f, arr[i], dx=dx))
    return ans


def count_conditional_probability(w1, t, w2, s, x, y):
    """已知条件随机场模型计算状态序列关于观测序列的非规范化条件概率

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param y: 需要计算的状态序列
    :return: 状态序列关于观测序列的条件概率
    """
    n_features_1 = len(w1)  # 转移特征数
    n_features_2 = len(w2)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    res = 0
    for k in range(n_features_1):
        for i in range(1, n_position):
            res += w1[k] * t[k](y[i - 1], y[i], x, i)
    for k in range(n_features_2):
        for i in range(n_position):
            res += w2[k] * s[k](y[i], x, i)
    return pow(math.e, res)


def make_hidden_sequence(w1, t, w2, s, x_range, y_range, n_samples=1000, random_state=0):
    """已知模型构造随机样本集

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x_range: 观测序列的可能取值
    :param y_range: 状态序列的可能取值
    :param n_samples: 生成样本集样本数(近似)
    :return: 状态序列关于观测序列的条件概率
    """
    P = [[0.0] * len(y_range) for _ in range(len(x_range))]  # 条件概率分布

    lst = []
    sum_ = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            P[i][j] = round(count_conditional_probability(w1, t, w2, s, x, y), 1)
            sum_ += P[i][j]
            lst.append(sum_)

    X, Y = [], []

    random.seed(random_state)
    for _ in range(n_samples):
        r = random.uniform(0, sum_)
        idx = bisect.bisect_left(lst, r)
        i, j = divmod(idx, len(y_range))
        X.append(x_range[i])
        Y.append(y_range[j])

    return X, Y


def improved_iterative_scaling(x, y, transfer_features, state_features, tol=1e-4, max_iter=1000):
    """改进的迭代尺度法学习条件随机场模型

    :param x: 输入变量
    :param y: 输出变量
    :param transfer_features: 转移特征函数
    :param state_features: 状态特征函数
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 条件随机场模型
    """
    n_samples = len(x)  # 样本数
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数

    # 坐标压缩
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 观测序列可能取值数
    n_y = len(y_list)  # 状态序列可能取值数
    n_total = n_x * n_y  # 观测序列可能取值和状态序列可能取值的笛卡尔积

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    # print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    # print("边缘分布的经验分布", d2)

    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features

    # 计算转移特征函数关于经验分布的期望值:EP(tk) (empirical_joint_distribution_each_feature)
    d3 = [0.0] * n_transfer_features  # empirical_joint_distribution_each_feature
    for k in range(n_transfer_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(1, n_position):
                    if transfer_features[k](y[i - 1], y[i], x, i):
                        d3[k] += d1[xi][yi]
                        nn[xi][yi] += 1

    # 计算状态特征函数关于经验分布的期望值:EP(sl) (empirical_joint_distribution_each_feature)
    d4 = [0.0] * n_state_features  # empirical_joint_distribution_each_feature
    for l in range(n_state_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(n_position):
                    if state_features[l](y[i], x, i):
                        d4[l] += d1[xi][yi]
                        nn[xi][yi] += 1

    # print("转移特征函数关于经验分布的期望值:", d3)
    # print("状态特征函数关于经验分布的期望值:", d4)
    # print("所有特征在(x,y)出现的次数:", nn)

    # 定义w的初值和模型P(Y|X)的初值
    w1 = [0] * n_transfer_features  # w的初值：wi=0
    w2 = [0] * n_state_features  # w的初值：wi=0
    p0 = [[1 / n_total] * n_y for _ in range(n_x)]  # 当wi=0时，P(Y|X)的值

    for _ in range(max_iter):
        change = False

        # 遍历所有转移特征以更新w
        for k in range(n_transfer_features):
            def func(d, kk):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        xx = x_list[xxi]
                        yy = y_list[yyi]
                        n_position = len(x_list[xxi])  # 序列中的位置数
                        val = 0
                        for i in range(1, n_position):
                            val += transfer_features[kk](yy[i - 1], yy[i], xx, i)
                        val *= d2[xxi] * p0[xxi][yyi] * pow(math.e, d * nn[xxi][yyi])
                        res += val
                res -= d3[kk]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(k,))

            # 更新wi的值
            w1[k] += dj
            if abs(dj) >= tol:
                change = True

        for l in range(n_state_features):
            def func(d, ll):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        xx = x_list[xxi]
                        yy = y_list[yyi]
                        n_position = len(x_list[xxi])  # 序列中的位置数
                        val = 0
                        for i in range(n_position):
                            val += state_features[ll](yy[i], xx, i)
                        val *= d2[xxi] * p0[xxi][yyi] * pow(math.e, d * nn[xxi][yyi])
                        res += val
                res -= d4[ll]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(l,))

            # 更新wi的值
            w2[l] += dj
            if abs(dj) >= tol:
                change = True

        # 计算新的模型
        p1 = [[0.0] * n_y for _ in range(n_x)]
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                res = 0
                for k in range(n_transfer_features):
                    for i in range(1, n_position):
                        res += w1[k] * t[k](y[i - 1], y[i], x, i)
                for l in range(n_state_features):
                    for i in range(n_position):
                        res += w2[l] * s[l](y[i], x, i)
                p1[xi][yi] = pow(math.e, res)
            total = sum(p1[xi][yi] for yi in range(n_y))
            if total > 0:
                for yi in range(n_y):
                    p1[xi][yi] /= total

        if not change:
            break

        p0 = p1

    ans = {}
    for xi in range(n_x):
        for yi in range(n_y):
            ans[(tuple(x_list[xi]), y_list[yi])] = p0[xi][yi]
    return w1 + w2, ans


def bfgs_algorithm(x, y, transfer_features, state_features, tol=1e-4, distance=20, max_iter=100):
    """拟牛顿法学习条件随机场模型

    :param x: 输入变量
    :param y: 输出变量
    :param transfer_features: 转移特征函数
    :param state_features: 状态特征函数
    :param tol: 容差
    :param distance: 一维搜索倍率
    :param max_iter: 最大迭代次数
    :return: 条件随机场模型
    """
    n_samples = len(x)  # 样本数
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数
    n_features = n_transfer_features + n_state_features  # 特征数

    # 坐标压缩
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 观测序列可能取值数
    n_y = len(y_list)  # 状态序列可能取值数
    n_total = n_x * n_y  # 观测序列可能取值和状态序列可能取值的笛卡尔积

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    # print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    # print("边缘分布的经验分布", d2)

    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features

    # 计算转移特征函数关于经验分布的期望值:EP(tk) (empirical_joint_distribution_each_feature)
    d3 = [0.0] * n_transfer_features  # empirical_joint_distribution_each_feature
    for k in range(n_transfer_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(1, n_position):
                    if transfer_features[k](y[i - 1], y[i], x, i):
                        d3[k] += d1[xi][yi]
                        nn[xi][yi] += 1

    # 计算状态特征函数关于经验分布的期望值:EP(sl) (empirical_joint_distribution_each_feature)
    d4 = [0.0] * n_state_features  # empirical_joint_distribution_each_feature
    for l in range(n_state_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(n_position):
                    if state_features[l](y[i], x, i):
                        d4[l] += d1[xi][yi]
                        nn[xi][yi] += 1

    # print("转移特征函数关于经验分布的期望值:", d3)
    # print("状态特征函数关于经验分布的期望值:", d4)
    # print("所有特征在(x,y)出现的次数:", nn)

    def func(ww):
        """目标函数"""
        res = 0
        for xxi in range(n_x):
            xx = x_list[xxi]
            n_position = len(xx)
            t1 = 0
            for yyi in range(n_y):
                yy = y_list[yyi]
                t2 = 0
                for kk in range(n_transfer_features):
                    for i in range(1, n_position):
                        if transfer_features[kk](yy[i - 1], yy[i], xx, i):
                            t2 += ww[kk]
                for ll in range(n_state_features):
                    for i in range(n_position):
                        if state_features[ll](yy[i], xx, i):
                            t2 += ww[ll + n_transfer_features]
                t1 += pow(math.e, t2)
            res += d2[xxi] * math.log(t1, math.e)

        for xxi in range(n_x):
            xx = x_list[xxi]
            n_position = len(xx)
            for yyi in range(n_y):
                yy = y_list[yyi]
                t3 = 0
                for kk in range(n_transfer_features):
                    for i in range(1, n_position):
                        if transfer_features[kk](yy[i - 1], yy[i], xx, i):
                            t3 += ww[kk]
                for ll in range(n_state_features):
                    for i in range(n_position):
                        if state_features[ll](yy[i], xx, i):
                            t3 += ww[ll + n_transfer_features]
                res -= d1[xxi][yyi] * t3

        return res

    # 定义w的初值和B0的初值
    w0 = [0] * n_features  # w的初值：wi=0
    B0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)

    for k in range(max_iter):
        print("迭代次数:", k, "->", [round(elem, 2) for elem in w0])

        # 计算梯度 gk
        nabla = partial_derivative(func, w0)

        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < tol:
            break

        # 计算pk
        if k == 0:
            pk = - B0 * g0  # 若numpy计算逆矩阵时有0，则对应位置会变为inf
        else:
            pk = - (B0 ** -1) * g0

        # 一维搜索求lambda_k
        def f(xx):
            """pk 方向的一维函数"""
            x2 = [w0[jj] + xx * float(pk[jj][0]) for jj in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # print(k, "lk:", lk)

        # 更新当前点坐标
        w1 = [w0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # print(k, "w1:", w1)

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        # 计算新的模型

        nabla = partial_derivative(func, w1)

        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < tol:
            w0 = w1
            break

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        # 当更新的距离小于精度要求时，停止迭代
        if pow(sum([lk * float(pk[j][0]) ** 2 for j in range(n_features)]), 0.5) < tol:
            w0 = w1
            break

        B1 = B0 + (yk * yk.T) / (yk.T * dk) + (B0 * dk * dk.T * B0) / (dk.T * B0 * dk)

        B0 = B1
        w0 = w1

    # 计算模型结果
    p0 = [[0.0] * n_y for _ in range(n_x)]
    for xi in range(n_x):
        for yi in range(n_y):
            x = x_list[xi]
            y = y_list[yi]
            n_position = len(x_list[xi])  # 序列中的位置数
            res = 0
            for k in range(n_transfer_features):
                for i in range(1, n_position):
                    res += w0[k] * t[k](y[i - 1], y[i], x, i)
            for l in range(n_state_features):
                for i in range(n_position):
                    res += w0[n_transfer_features + l] * s[l](y[i], x, i)
            p0[xi][yi] = pow(math.e, res)
        total = sum(p0[xi][yi] for yi in range(n_y))
        if total > 0:
            for yi in range(n_y):
                p0[xi][yi] /= total

    # 返回最终结果
    ans = {}
    for xi in range(n_x):
        for yi in range(n_y):
            ans[(tuple(x_list[xi]), y_list[yi])] = p0[xi][yi]
    return w0, ans


def viterbi_algorithm(w1, transfer_features, w2, state_features, x, n_state):
    """维特比算法预测状态序列

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param n_state: 状态的可能取值数
    :return: 最优可能的状态序列
    """
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    # 定义状态矩阵
    dp = [[0.0] * n_state for _ in range(n_position)]  # 概率最大值
    last = [[-1] * n_state for _ in range(n_position)]  # 上一个结点

    # 处理t=0的情况
    for i in range(n_state):
        for l in range(n_state_features):
            dp[0][i] += w2[l] * state_features[l](y0=i, x=x, i=0)

    # 处理t>0的情况
    for t in range(1, n_position):
        for i in range(n_state):
            for j in range(n_state):
                d = dp[t - 1][i]
                for k in range(n_transfer_features):
                    d += w1[k] * transfer_features[k](y0=i, y1=j, x=x, i=t)
                for l in range(n_state_features):
                    d += w2[l] * state_features[l](y0=j, x=x, i=t)
                # print((i, j), "=", d)
                if d >= dp[t][j]:
                    dp[t][j] = d
                    last[t][j] = i
        # print(dp[t], last[t])

    # 计算最优路径的终点
    best_end, best_gamma = 0, 0
    for i in range(n_state):
        if dp[-1][i] > best_gamma:
            best_end, best_gamma = i, dp[-1][i]

    # 计算最优路径
    ans = [0] * (n_position - 1) + [best_end]
    for t in range(n_position - 1, 0, -1):
        ans[t - 1] = last[t][ans[t]]
    return ans


if __name__ == "__main__":
    print("开始测试模型计算条件概率……")

    def t1(y0, y1, x, i):
        return int(y0 in {0} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)} and i in {1, 2})

    def t2(y0, y1, x, i):
        return int(y0 in {0} and y1 in {0} and x in {(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {1})

    def t3(y0, y1, x, i):
        return int(y0 in {1} and y1 in {0, 1} and x in {(0, 0, 0), (1, 1, 1)} and i in {2})

    def t4(y0, y1, x, i):
        return int(y0 in {1} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {2})

    def t5(y0, y1, x, i):
        return int(y0 in {0, 1} and y1 in {0} and x in {(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)} and i in {1, 2})

    def s1(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1, 2})

    def s2(y0, x, i):
        return int(y0 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {0})

    def s3(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1})

    def s4(y0, x, i):
        return int(y0 in {1} and x in {(1, 0, 1), (0, 1, 0)} and i in {0, 2})

    w1 = [1, 0.6, 1.2, 0.2, 1.4]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.2, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    for x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)}:
        for y in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)}:
            print(x, "->", y, ":", count_conditional_probability(w1, t, w2, s, x, y))

    print("------------------------------------------")
    print("开始测试模型构造随机样本集……")
    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])

    for row in zip(X, Y):
        print(row)

    print("------------------------------------------")
    print("开始测试条件随机场模型学习的改进的迭代尺度法……")
    # 生成随机模型
    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
    w, P = improved_iterative_scaling(X, Y, t, s)
    print("学习结果:", [round(elem, 2) for elem in w])

    print("------------------------------------------")
    print("开始测试条件随机场模型学习的BFGS算法……")
    # 生成随机模型
    w1 = [1, 0.6, 1.2, 0.2, 1.4]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.2, 0.8, 0.5]
    s = [s1, s2, s3, s4]
    # 生成随机模型
    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
    w, P = bfgs_algorithm(X, Y, t, s)
    print("学习结果:", [round(elem, 2) for elem in w])

    print("------------------------------------------")
    print("开始测试条件随机场预测的维特比算法……")

    def t1(y0, y1, x, i):
        return int(y0 == 0 and y1 == 1 and i in {1, 2})

    def t2(y0, y1, x, i):
        return int(y0 == 0 and y1 == 0 and i in {1})

    def t3(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {2})

    def t4(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {1})

    def t5(y0, y1, x, i):
        return int(y0 == 1 and y1 == 1 and i in {2})

    def s1(y0, x, i):
        return int(y0 == 0 and i in {0})

    def s2(y0, x, i):
        return int(y0 == 1 and i in {0, 1})

    def s3(y0, x, i):
        return int(y0 == 0 and i in {1, 2})

    def s4(y0, x, i):
        return int(y0 == 1 and i in {2})
    w1 = [1, 0.6, 1, 1, 0.2]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.5, 0.8, 0.5]
    s = [s1, s2, s3, s4]
    print(viterbi_algorithm(w1, t, w2, s, [random.randint(0, 1) for _ in range(3)], 2))  # [0, 1, 0]
