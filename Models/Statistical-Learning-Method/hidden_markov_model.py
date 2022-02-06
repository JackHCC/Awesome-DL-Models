#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :hidden_markov_model.py
@Author  :JackHCC
@Date    :2022/2/6 15:23 
@Desc    :Implement HMM forward and backward, Viterbi Algorithm

'''
import numpy as np


# HMM评估-后向算法
class HiddenMarkovBackward:
    def __init__(self, verbose=False):
        self.betas = None
        self.backward_P = None
        self.verbose = verbose

    def backward(self, Q, V, A, B, O, PI):
        """
        后向算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # (1)初始化后向概率beta值，书中第201页公式(10.19)
        betas = np.ones((N, M))
        if self.verbose:
            self.print_betas_T(N, M)

        # (2)对观测序列逆向遍历，M-2即为T-1
        if self.verbose:
            print("\n从时刻T-1到1观测序列的后向概率：")
        for t in range(M - 2, -1, -1):
            # 得到序列对应的索引
            index_of_o = V.index(O[t + 1])
            # 遍历状态序列
            for i in range(N):
                # 书中第201页公式(10.20)
                betas[i][t] = np.dot(np.multiply(A[i], [b[index_of_o] for b in B]),
                                     [beta[t + 1] for beta in betas])
                real_t = t + 1
                real_i = i + 1
                if self.verbose:
                    self.print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t)

        # 取出第一个值索引，用于得到o1
        index_of_o = V.index(O[0])
        self.betas = betas
        # 书中第201页公式(10.21)
        P = np.dot(np.multiply(PI, [b[index_of_o] for b in B]),
                   [beta[0] for beta in betas])
        self.backward_P = P
        self.print_P(B, N, P, PI, betas, index_of_o)

    @staticmethod
    def print_P(B, N, P, PI, betas, index_of_o):
        print("\n观测序列概率：")
        print("P(O|lambda) = ", end="")
        for i in range(N):
            print("%.1f * %.1f * %.5f + "
                  % (PI[0][i], B[i][index_of_o], betas[i][0]), end="")
        print("0 = %f" % P)

    @staticmethod
    def print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t):
        print("beta%d(%d) = sum[a%dj * bj(o%d) * beta%d(j)] = ("
              % (real_t, real_i, real_i, real_t + 1, real_t + 1), end='')
        for j in range(N):
            print("%.2f * %.2f * %.2f + "
                  % (A[i][j], B[j][index_of_o], betas[j][t + 1]), end='')
        print("0) = %.3f" % betas[i][t])

    @staticmethod
    def print_betas_T(N, M):
        print("初始化后向概率：")
        for i in range(N):
            print('beta%d(%d) = 1' % (M, i + 1))


# HMM评估-前向算法
class HiddenMarkovForwardBackward(HiddenMarkovBackward):
    def __init__(self, verbose=False):
        super(HiddenMarkovBackward, self).__init__()
        self.alphas = None
        self.forward_P = None
        self.verbose = verbose

    def forward(self, Q, V, A, B, O, PI):
        """
        前向算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化前向概率alpha值
        alphas = np.zeros((N, M))
        # 时刻数=观测序列数
        T = M
        # (2)对观测序列遍历，遍历每一个时刻，计算前向概率alpha值

        for t in range(T):
            if self.verbose:
                if t == 0:
                    print("前向概率初值：")
                elif t == 1:
                    print("\n从时刻1到T-1观测序列的前向概率：")
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                if t == 0:
                    # (1)初始化alpha初值，书中第198页公式(10.15)
                    alphas[i][t] = PI[t][i] * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t1(alphas, i, t)
                else:
                    # (2)递推，书中第198页公式(10.16)
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas],
                                          [a[i] for a in A]) * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t(alphas, i, t)
        # (3)终止，书中第198页公式(10.17)
        self.forward_P = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas

    @staticmethod
    def print_alpha_t(alphas, i, t):
        print("alpha%d(%d) = [sum alpha%d(i) * ai%d] * b%d(o%d) = %f"
              % (t + 1, i + 1, t - 1, i, i, t, alphas[i][t]))

    @staticmethod
    def print_alpha_t1(alphas, i, t):
        print('alpha1(%d) = pi%d * b%d * b(o1) = %f'
              % (i + 1, i, i, alphas[i][t]))

    def calc_t_qi_prob(self, t, qi):
        result = (self.alphas[qi - 1][t - 1] * self.betas[qi - 1][t - 1]) / self.backward_P[0]
        if self.verbose:
            print("计算P(i%d=q%d|O,lambda)：" % (t, qi))
            print("P(i%d=q%d|O,lambda) = alpha%d(%d) * beta%d(%d) / P(O|lambda) = %f"
                  % (t, qi, t, qi, t, qi, result))

        return result


# HMM解码：维特比算法
class HiddenMarkovViterbi:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def viterbi(self, Q, V, A, B, O, PI):
        """
        维特比算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化deltas
        deltas = np.zeros((N, M))
        # 初始化psis
        psis = np.zeros((N, M))

        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1, M))
        # (2)递推，遍历观测序列
        for t in range(M):
            if self.verbose:
                if t == 0:
                    print("初始化Psi1和delta1：")
                elif t == 1:
                    print("\n从时刻2到T的所有单个路径中概率最大值delta和概率最大的路径的第t-1个结点Psi：")

            # (2)递推从t=2开始
            real_t = t + 1
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            for i in range(N):
                real_i = i + 1
                if t == 0:
                    # (1)初始化
                    deltas[i][t] = PI[0][i] * B[i][index_of_o]
                    psis[i][t] = 0

                    self.print_delta_t1(B, PI, deltas, i, index_of_o, real_i, t)
                    self.print_psi_t1(real_i)
                else:
                    # (2)递推，对t=2,3,...,T
                    deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas],
                                                      [a[i] for a in A])) * B[i][index_of_o]
                    self.print_delta_t(A, B, deltas, i, index_of_o, real_i, real_t, t)

                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas],
                                                       [a[i] for a in A]))
                    self.print_psi_t(i, psis, real_i, real_t, t)

        last_deltas = [delta[M - 1] for delta in deltas]
        # (3)终止，得到所有路径的终结点最大的概率值
        P = np.max(last_deltas)
        # (3)得到最优路径的终结点
        I[0][M - 1] = np.argmax(last_deltas)
        if self.verbose:
            print("\n所有路径的终结点最大的概率值：")
            print("P = %f" % P)
        if self.verbose:
            print("\n最优路径的终结点：")
            print("i%d = argmax[deltaT(i)] = %d" % (M, I[0][M - 1] + 1))
            print("\n最优路径的其他结点：")

        # (4)递归由后向前得到其他结点
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            if self.verbose:
                print("i%d = Psi%d(i%d) = %d" % (t + 1, t + 2, t + 2, I[0][t] + 1))

        # 输出最优路径
        print("\n最优路径是：", "->".join([str(int(i + 1)) for i in I[0]]))

    def print_psi_t(self, i, psis, real_i, real_t, t):
        if self.verbose:
            print("Psi%d(%d) = argmax[delta%d(j) * aj%d] = %d"
                  % (real_t, real_i, real_t - 1, real_i, psis[i][t]))

    def print_delta_t(self, A, B, deltas, i, index_of_o, real_i, real_t, t):
        if self.verbose:
            print("delta%d(%d) = max[delta%d(j) * aj%d] * b%d(o%d) = %.2f * %.2f = %.5f"
                  % (real_t, real_i, real_t - 1, real_i, real_i, real_t,
                     np.max(np.multiply([delta[t - 1] for delta in deltas],
                                        [a[i] for a in A])),
                     B[i][index_of_o], deltas[i][t]))

    def print_psi_t1(self, real_i):
        if self.verbose:
            print("Psi1(%d) = 0" % real_i)

    def print_delta_t1(self, B, PI, deltas, i, index_of_o, real_i, t):
        if self.verbose:
            print("delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f"
                  % (real_i, real_i, real_i, PI[0][i], B[i][index_of_o], deltas[i][t]))


if __name__ == '__main__':
    print("开始测试HMM评估-前向与后向算法……")
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    PI = [[0.2, 0.3, 0.5]]

    hmm_forward_backward = HiddenMarkovForwardBackward(verbose=True)
    hmm_forward_backward.forward(Q, V, A, B, O, PI)
    print()
    hmm_forward_backward.backward(Q, V, A, B, O, PI)
    print()
    hmm_forward_backward.calc_t_qi_prob(t=4, qi=3)

    print("------------------------------------------")
    print("开始测试HMM解码-维特比算法……")
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '白']
    PI = [[0.2, 0.4, 0.4]]

    HMM = HiddenMarkovViterbi(verbose=True)
    HMM.viterbi(Q, V, A, B, O, PI)
