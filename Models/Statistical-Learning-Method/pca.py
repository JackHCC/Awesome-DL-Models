#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :pca.py
@Author  :JackHCC
@Date    :2022/2/8 13:17 
@Desc    :Implement PCA

'''
import numpy as np
import pandas as pd


# 根据保留多少维特征进行降维
class PCAcomponent(object):
    def __init__(self, X, N=3):
        self.X = X
        self.N = N
        self.variance_ratio = []
        self.low_dataMat = []

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 另一种计算协方差矩阵的方法：dataMat.T * dataMat / dataMat.shape[0]
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        self.low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵
        # reconMat = (low_dataMat * small_eigVect.I) + X_mean  # 重构数据
        # 输出每个维度所占的方差百分比
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self


# 根据保留多大方差百分比进行降维
class PCApercent(object):
    def __init__(self, X, percentage=0.95):
        self.X = X
        self.percentage = percentage
        self.variance_ratio = []
        self.low_dataMat = []

    # 通过方差百分比选取前n个主成份
    def percent2n(self, eigVal):
        sortVal = np.sort(eigVal)[-1::-1]
        percentSum, componentNum = 0, 0
        for i in sortVal:
            percentSum += i
            componentNum += 1
            if percentSum >= sum(sortVal) * self.percentage:
                break
        return componentNum

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        covMat = np.cov(dataMat, rowvar=False)
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        n = self.percent2n(eigVal)
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(n + 1):-1]
        n_eigVect = eigVect[:, eigValInd]
        self.low_dataMat = dataMat * n_eigVect
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self


if __name__ == "__main__":
    print("开始测试PCA算法……")
    df = pd.read_csv(r'./data/iris.data', header=None)
    data, label = df[range(len(df.columns) - 1)], df[[len(df.columns) - 1]]
    data = np.mat(data)
    print("Original dataset = {}*{}".format(data.shape[0], data.shape[1]))
    pca = PCAcomponent(data, 2)
    # pca = PCApercent(data, 0.98)
    pca.fit()
    print(pca.low_dataMat)
    print(pca.variance_ratio)
