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


def pca_by_feature(R, need_accumulative_contribution_rate=0.75):
    """协方差矩阵/相关矩阵求解主成分及其因子载荷量和贡献率（打印到控制台）

    :param R: 协方差矩阵/相关矩阵
    :param need_accumulative_contribution_rate: 需要达到的累计方差贡献率
    :return: None
    """
    n_features = len(R)

    # 求解相关矩阵的特征值和特征向量
    features_value, features_vector = np.linalg.eig(R)

    # 依据特征值大小排序特征值和特征向量
    z = [(features_value[i], features_vector[:, i]) for i in range(n_features)]
    z.sort(key=lambda x: x[0], reverse=True)
    features_value = [z[i][0] for i in range(n_features)]
    features_vector = np.hstack([z[i][1][:, np.newaxis] for i in range(n_features)])

    # 计算所需的主成分数量
    total_features_value = sum(features_value)  # 特征值总和
    need_accumulative_contribution_rate *= total_features_value
    n_principal_component = 0  # 所需的主成分数量
    accumulative_contribution_rate = 0
    while accumulative_contribution_rate < need_accumulative_contribution_rate:
        accumulative_contribution_rate += features_value[n_principal_component]
        n_principal_component += 1

    # 输出单位特征向量和主成分的方差贡献率
    print("【单位特征向量和主成分的方差贡献率】")
    for i in range(n_principal_component):
        print("主成分:", i,
              "方差贡献率:", features_value[i] / total_features_value,
              "特征向量:", features_vector[:, i])

    # 计算各个主成分的因子载荷量：factor_loadings[i][j]表示第i个主成分对第j个变量的相关关系，即因子载荷量
    factor_loadings = np.vstack(
        [[np.sqrt(features_value[i]) * features_vector[j][i] / np.sqrt(R[j][j]) for j in range(n_features)]
         for i in range(n_principal_component)]
    )

    # 输出主成分的因子载荷量和贡献率
    print("【主成分的因子载荷量和贡献率】")
    for i in range(n_principal_component):
        print("主成分:", i, "因子载荷量:", factor_loadings[i])
    print("所有主成分对变量的贡献率:", [np.sum(factor_loadings[:, j] ** 2) for j in range(n_features)])


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
    print("开始测试相关矩阵求解主成分及其因子载荷量和贡献率……")
    X = np.array([[1, 0.44, 0.29, 0.33],
                  [0.44, 1, 0.35, 0.32],
                  [0.29, 0.35, 1, 0.60],
                  [0.33, 0.32, 0.60, 1]])
    pca_by_feature(X)

    print("------------------------------------------")
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
