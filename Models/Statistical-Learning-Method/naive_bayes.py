#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :naive_bayes.py
@Author  :JackHCC
@Date    :2022/2/10 14:35 
@Desc    :Implement Naive Bayes

'''
import collections


class NaiveBayesAlgorithmHashmap:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数
        self.n = len(x[0])  # 维度数

        count1 = collections.Counter(y)  # 先验概率的分子，条件概率的分母
        count2 = [collections.Counter() for _ in range(self.n)]  # 条件概率的分子
        for i in range(self.N):
            for j in range(self.n):
                count2[j][(x[i][j], y[i])] += 1

        # 计算先验概率和条件概率
        self.prior = {k: v / self.N for k, v in count1.items()}
        self.conditional = [{k: v / count1[k[1]] for k, v in count2[j].items()} for j in range(self.n)]

    def predict(self, x):
        best_y, best_score = 0, 0
        for y in self.prior:
            score = self.prior[y]
            for j in range(self.n):
                score *= self.conditional[j][(x[j], y)]
            if score > best_score:
                best_y, best_score = y, score
        return best_y


class NaiveBayesAlgorithmArray:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        table1 = [0] * self.K
        for i in range(self.N):
            table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = table1[k] / self.N

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = table2[j][k][t] / table1[k]

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= 0
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y


class NaiveBayesAlgorithmWithSmoothing:
    """贝叶斯估计（仅支持离散型数据）"""

    def __init__(self, x, y, l=1):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数
        self.l = l  # 贝叶斯估计的lambda参数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        self.table1 = [0] * self.K
        for i in range(self.N):
            self.table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        self.table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                self.table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = (self.table1[k] + self.l) / (self.N + self.l * self.K)

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = (self.table2[j][k][t] + self.l) / (self.table1[k] + self.l * self.Sj[j])

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= self.l / (self.table1[k] + self.l * self.Sj[j])
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y


if __name__ == "__main__":
    print("开始测试朴素贝叶斯算法……")
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes_1 = NaiveBayesAlgorithmHashmap(*dataset)
    print(naive_bayes_1.predict([2, "S"]))
    naive_bayes_2 = NaiveBayesAlgorithmArray(*dataset)
    print(naive_bayes_2.predict([2, "S"]))

    print("------------------------------------------")
    print("开始测试贝叶斯估计……")
    naive_bayes = NaiveBayesAlgorithmWithSmoothing(*dataset)
    print(naive_bayes.predict([2, "S"]))

    """
    支持连续型特征的朴素贝叶斯可以使用sklearn库
    
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> from sklearn.naive_bayes import ComplementNB
    >>> from sklearn.naive_bayes import BernoulliNB
    
    >>> X, Y = load_breast_cancer(return_X_y=True)
    >>> x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
    
    >>> # 高斯朴素贝叶斯
    >>> gnb = GaussianNB()
    >>> gnb.fit(x1, y1)
    >>> gnb.score(x2, y2)
    0.9210526315789473
    
    >>> # 多项分布朴素贝叶斯
    >>> mnb = MultinomialNB()
    >>> mnb.fit(x1, y1)
    >>> mnb.score(x2, y2)
    0.9105263157894737
    
    >>> # 补充朴素贝叶斯
    >>> mnb = ComplementNB()
    >>> mnb.fit(x1, y1)
    >>> mnb.score(x2, y2)
    0.9052631578947369
    
    >>> # 伯努利朴素贝叶斯
    >>> bnb = BernoulliNB()
    >>> bnb.fit(x1, y1)
    >>> bnb.score(x2, y2)
    0.6421052631578947
    """