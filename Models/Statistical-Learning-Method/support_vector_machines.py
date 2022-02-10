#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :support_vector_machines.py
@Author  :JackHCC
@Date    :2022/2/6 12:21 
@Desc    :Support Vector Machine using the Sequential Minimal Optimization (SMO) algorithm for training.

'''
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class SVM:
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(self, max_iter=10000, kernel_type='linear', C=10000.0, epsilon=0.001):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n - 1, j)  # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)


def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)


class SVMSmo:
    """支持向量机, SMO算法另一种实现

    :param X: 输入变量列表
    :param Y: 输出变量列表
    :param C: 正则化项（惩罚参数：C越大，对误分类的惩罚越大）
    :param kernel_func: 核函数
    :param tol: 容差
    :param max_iter: 最大迭代次数
    """

    def __init__(self, X, Y, kernel_func=None, C=1, tol=1e-4, max_iter=100):
        # ---------- 检查参数 ----------
        # 检查输入变量和输出变量
        if len(X) != len(Y):
            raise ValueError("输入变量和输出变量的样本数不同")
        if len(X) == 0:
            raise ValueError("输入样本数不能为0")
        self.X, self.Y = X, Y

        # 检查正则化项
        if C <= 0:
            raise ValueError("正则化项必须严格大于0")
        self.C = C

        # 检查核函数
        if kernel_func is None:
            kernel_func = self._linear_kernel  # 当未设置核函数时默认使用线性核函数
        self.kernel_func = kernel_func

        # 检查容差
        if tol <= 0:
            raise ValueError("容差必须大于0")
        self.tol = tol

        # 检查最大迭代步数
        if max_iter <= 0:
            raise ValueError("迭代步数必须大于0")
        self.max_iter = max_iter

        # ---------- 初始化计算 ----------
        self.n_samples = len(X)  # 计算样本数
        self.n_features = len(X[0])  # 计算特征数
        self.kernel_matrix = self._count_kernel_matrix()  # 计算核矩阵

        # ---------- 取初值 ----------
        self.A = np.zeros(self.n_samples)  # 拉格朗日乘子(alpha)
        self.b = 0  # 参数b
        self.E = [float(-self.Y[i]) for i in range(self.n_samples)]  # 初始化Ei的列表

        # ---------- SMO算法训练支持向量机 ----------
        self.smo()  # SMO算法计算了拉格朗日乘子的近似解
        self.support = [i for i, v in enumerate(self.A) if v > 0]  # 计算支持向量的下标列表

    def smo(self):
        """使用序列最小最优化(SMO)算法训练支持向量机"""
        for k in range(self.max_iter):
            change_num = 0  # 更新的样本数

            for i1 in self.outer_circle():  # 外层循环：依据7.4.2.1选择第1个变量（找到a1并更新后继续向后遍历，而不回到第1个）
                i2 = next(self.inner_circle(i1))  # 内层循环：依据7.4.2.2选择第2个变量（没有处理特殊情况下用启发式规则继续寻找a2）

                a1_old, a2_old = self.A[i1], self.A[i2]
                y1, y2 = self.Y[i1], self.Y[i2]
                k11, k22, k12 = self.kernel_matrix[i1][i1], self.kernel_matrix[i2][i2], self.kernel_matrix[i1][i2]

                eta = k11 + k22 - 2 * k12  # 根据式(7.107)计算η(eta)
                a2_new = a2_old + y2 * (self.E[i1] - self.E[i2]) / eta  # 依据式(7.106)计算未经剪辑的a2_new

                # 计算a2_new所在对角线线段端点的界
                if y1 != y2:
                    l = max(0, a2_old - a1_old)
                    h = min(self.C, self.C + a2_old - a1_old)
                else:
                    l = max(0, a2_old + a1_old - self.C)
                    h = min(self.C, a2_old + a1_old)

                # 依据式(7.108)剪辑a2_new
                if a2_new > h:
                    a2_new = h
                if a2_new < l:
                    a2_new = l

                # 依据式(7.109)计算a_new
                a1_new = a1_old + y1 * y2 * (a2_old - a2_new)

                # 依据式(7.115)和式(7.116)计算b1_new和b2_new并更新b
                b1_new = -self.E[i1] - y1 * k11 * (a1_new - a1_old) - y2 * k12 * (a2_new - a2_old) + self.b
                b2_new = -self.E[i2] - y1 * k12 * (a1_new - a1_old) - y2 * k22 * (a2_new - a2_old) + self.b
                if 0 < a1_new < self.C and 0 < a2_new < self.C:
                    self.b = b1_new
                else:
                    self.b = (b1_new + b2_new) / 2

                # 更新a1,a2
                self.A[i1], self.A[i2] = a1_new, a2_new

                # 依据式(7.105)计算并更新E
                self.E[i1], self.E[i2] = self._count_g(i1) - y1, self._count_g(i2) - y2

                if abs(a2_new - a2_old) > self.tol:
                    change_num += 1

            print("迭代次数:", k, "change_num =", change_num)

            if change_num == 0:
                break

    def predict(self, x):
        """预测实例"""
        return np.sign(sum(self.A[i] * self.Y[i] * self.kernel_func(x, self.X[i]) for i in self.support) + self.b)

    def _linear_kernel(self, x1, x2):
        """计算特征向量x1和特征向量x2的线性核函数的值"""
        return sum(x1[i] * x2[i] for i in range(self.n_features))

    def outer_circle(self):
        """外层循环生成器"""
        for i1 in range(self.n_samples):  # 先遍历所有在间隔边界上的支持向量点
            if -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1
        for i1 in range(self.n_samples):  # 再遍历整个训练集的所有样本点
            if not -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1

    def inner_circle(self, i1):
        """内层循环生成器：未考虑特殊情况下启发式选择a2的情况"""
        max_differ = 0
        i2 = -1
        for ii2 in range(self.n_samples):
            differ = abs(self.E[i1] - self.E[ii2])
            if differ > max_differ:
                i2, max_differ = ii2, differ
        yield i2

    def _count_kernel_matrix(self):
        """计算核矩阵"""
        kernel_matrix = [[0] * self.n_samples for _ in range(self.n_samples)]
        for i1 in range(self.n_samples):
            for i2 in range(i1, self.n_samples):
                kernel_matrix[i1][i2] = kernel_matrix[i2][i1] = self.kernel_func(self.X[i1], self.X[i2])
        return kernel_matrix

    def _count_g(self, i1):
        """依据式(7.104)计算g(x)"""
        return sum(self.A[i2] * self.Y[i2] * self.kernel_matrix[i1][i2] for i2 in range(self.n_samples)) + self.b

    def _satisfied_kkt(self, i):
        """判断是否满足KKT条件"""
        ygi = self.Y[i] * self._count_g(i)  # 计算 yi*g(xi)
        if -self.tol < self.A[i] < self.tol and ygi >= 1 - self.tol:
            return True  # (7.111)式的情况: ai=0 && yi*g(xi)>=1
        elif -self.tol < self.A[i] < self.C + self.tol and abs(ygi - 1) < self.tol:
            return True  # (7.112)式的情况: 0<ai<C && yi*g(xi)=1
        elif self.C - self.tol < self.A[i] < self.C + self.tol and ygi <= 1 + self.tol:
            return True  # (7.113)式的情况: ai=C && yi*g(xi)<=1
        else:
            return False


if __name__ == "__main__":
    print("开始测试SVM算法……")
    # 加载数据
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])

    # Initialize model
    model = SVM()

    # Fit model
    support_vectors, iterations = model.fit(X, y)

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("Converged after %d iterations" % (iterations))

    # 绘制数据点
    color_seq = ['red' if v == 1 else 'blue' for v in y]
    plt.scatter([i[0] for i in X], [i[1] for i in X], c=color_seq)
    # 得到x轴的所有点
    xaxis = np.linspace(0, 3.5)
    w = model.w
    # 计算斜率
    a = -w[0] / w[1]
    # 得到分离超平面
    y_sep = a * xaxis - (model.b) / w[1]
    # 下边界超平面
    b = support_vectors[0]
    yy_down = a * xaxis + (b[1] - a * b[0])
    # 上边界超平面
    b = support_vectors[-1]
    yy_up = a * xaxis + (b[1] - a * b[0])
    # 绘制超平面
    plt.plot(xaxis, y_sep, 'k-')
    plt.plot(xaxis, yy_down, 'k--')
    plt.plot(xaxis, yy_up, 'k--')
    # 绘制支持向量
    plt.xlabel('$x^{(1)}$')
    plt.ylabel('$x^{(2)}$')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                s=150, facecolors='none', edgecolors='k')
    plt.show()
