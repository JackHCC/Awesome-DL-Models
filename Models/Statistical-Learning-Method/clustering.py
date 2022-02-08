#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :clustering.py
@Author  :JackHCC
@Date    :2022/2/8 12:13 
@Desc    :Implement Hierarchical Clustering and K-means

'''

import numpy as np
import math
import time
import random
from scipy.special import comb
from data.utils import load_data
import matplotlib.pyplot as plt


# 定义标准化函数，对每一列特征进行min-max标准化，将数据缩放到0-1之间
# 标准化处理对于计算距离的机器学习方法是非常重要的，因为特征的尺度不同会导致计算出来的距离倾向于尺度大的特征，为保证距离对每一列特征都是公平的，必须将所有特征缩放到同一尺度范围内
def normalize(Xarray):
    '''
    INPUT:
    Xarray - (array) 特征数据数组

    OUTPUT:
    Xarray - (array) 标准化处理后的特征数据数组

    '''
    for f in range(Xarray.shape[1]):
        maxf = np.max(Xarray[:, f])
        minf = np.min(Xarray[:, f])
        for n in range(Xarray.shape[0]):
            Xarray[n][f] = (Xarray[n][f] - minf) / (maxf - minf)
    return Xarray


# 定义计算两条数据间的距离的函数，这里计算的是欧式距离
def cal_distance(xi, xj):
    '''
    INPUT:
    Xi - (array) 第i条特征数据
    Xj - (array) 第j条特征数据

    OUTPUT:
    dist - (float) 两条数据的欧式距离

    '''
    dist = 0
    for col in range(len(xi)):
        dist += (xi[col] - xj[col]) ** 2
    dist = math.sqrt(dist)
    return dist


# 定义计算所有特征数据两两之间距离的函数
def distances(Xarray):
    '''
    INPUT:
    Xarray - (array) 特征数据数组

    OUTPUT:
    dists - (array) 两两数据的欧式距离数组

    '''
    dists = np.zeros((Xarray.shape[0], Xarray.shape[0]))  # 定义一个数组用来保存两两数据的距离
    for n1 in range(Xarray.shape[0]):
        for n2 in range(n1):
            dists[n1][n2] = cal_distance(Xarray[n1], Xarray[n2])
            dists[n2][n1] = dists[n1][n2]
    return dists


# 定义计算两类的类间距离的函数，这里计算的是最短距离
def cal_group_dist(g1, g2, group_dict, dists):
    '''
    INPUT:
    g1 - (int) 类别1的标签
    g2 - (int) 类别2的标签
    group_dict - (dict) 类别字典
    dists - (array) 两两数据的欧式距离数组

    OUTPUT:
    (int) 类间最短距离

    '''
    d = []
    # 循环计算两类之间两两数据的距离
    for xi in group_dict[g1]:
        for xj in group_dict[g2]:
            if xi != xj:
                d.append(dists[xi][xj])
    return min(d)


# 定义层次聚类函数
def hierarchical_clustering(Xarray, k, dists):
    '''
    INPUT:
    Xarray - (array) 特征数据数组
    k - (int) 设定的类别数
    dists - (array) 两两数据的欧式距离数组

    OUTPUT:
    group_dict - (dict) 类别字典

    '''
    group_dict = {}  # 定义一个空字典，用于保存聚类所产生的所有类别
    for n in range(Xarray.shape[0]):  # 层次聚类是一种聚合聚类方法，首先将每条数据都分到不同的类，数据的类别标签为0-(N-1)，其中N为数据条数
        group_dict[n] = [n]
    newgroup = Xarray.shape[0]  # newgroup表示新的类别标签，此时下一个类别标签为N
    while len(group_dict.keys()) > k:  # 当类别数大于我们所设定的类别数k时，不断循环进行聚类
        print('Number of groups:', len(group_dict.keys()))
        group_dists = {}  # 定义一个空字典，用于保存两两类之间的间距，其中字典的值为元组(g1, g2)，表示两个类别标签，字典的键为这两个类别的间距
        # 循环计算group_dict中两两类别之间的间距，保存到group_dists中
        for g1 in group_dict.keys():
            for g2 in group_dict.keys():
                if g1 != g2:
                    if (g1, g2) not in group_dists.values():
                        d = cal_group_dist(g1, g2, group_dict, dists)
                        group_dists[d] = (g1, g2)
        group_mindist = min(list(group_dists.keys()))  # 取类别之间的最小间距
        mingroups = group_dists[group_mindist]  # 取间距最小的两个类别
        new = []  # 定义一个列表，用于保存所产生的新类中包含的数据，这里用之前对每条数据给的类别标签0-(N-1)来表示
        for g in mingroups:
            new.extend(group_dict[g])  # 将间距最小的两类中包含的数据保存在new列表中
            del group_dict[g]  # 然后在group_dict中移去这两类
        print(newgroup, new)
        group_dict[newgroup] = new  # 此时聚类所产生的新类中包含的数据即为以上两类的中包含的数据的聚合，给新类贴上类别标签为newgroup，保存到group_dict中
        newgroup += 1  # 产生下一个类别标签
    return group_dict


# 定义计算类中心的函数，以当前类中所包含数据的各个特征均值作为新的新的类中心
def cal_group_center(group, Xarray):
    '''
    INPUT:
    group - (list) 类所包含的数据列表
    Xarray - (array) 特征数据数组

    OUTPUT:
    center - (array) 新的类中心

    '''
    center = np.zeros(Xarray.shape[1])
    for i in range(Xarray.shape[1]):
        for n in group:
            center[i] += Xarray[n][i]  # 计算当前类中第i个特征的数据之和
    center = center / Xarray.shape[0]  # 计算各个特征的均值
    return center


# 定义k均值聚类函数
def k_means(Xarray, k, iters):
    '''
    INPUT:
    Xarray - (array) 特征数据数组
    k - (int) 设定的类别数
    iters - (int) 设定的迭代次数

    OUTPUT:
    group_dict - (dict) 类别字典
    scores - (int) 每次迭代的ARI得分列表

    '''
    center_inds = random.sample(range(Xarray.shape[0]), k)  # 从特征数据中随机抽取k个数据索引
    centers = [Xarray[ci] for ci in center_inds]  # 将这k个数据索引所对应的特征数据作为初始的k个聚类中心
    scores = []  # 定义一个空列表用来保存每次迭代的ARI得分
    for i in range(iters):
        group_dict = {i: [] for i in range(k)}  # 定义一个空字典，用于保存聚类所产生的所有类别，其中字典的键为类别标签，值为类别所包含的数据列表，以索引表示每条数据
        print('{}/{}'.format(i + 1, iters))
        # 循环计算每条数据到各个聚类中心的距离
        for n in range(Xarray.shape[0]):
            dists = []  # 保存第n条数据到各个聚类中心的距离
            for ci in range(k):
                dist = cal_distance(Xarray[n], centers[ci])
                dists.append(dist)
            g = dists.index(min(dists))  # 取距离最近的中心所在的类
            group_dict[g].append(n)  # 将第n条数据的索引n保存到g类
        print(group_dict)
        for i in range(k):
            centers[i] = cal_group_center(group_dict[i], Xarray)  # 根据每一类所包含的数据重新计算类中心
        scores.append(adjusted_rand_index(group_dict, Ylist, k))  # 将该轮迭代的ARI得分保存到scores列表
    return group_dict, scores


# 定义计算调整兰德系数(ARI)的函数，调整兰德系数是一种聚类方法的常用评估方法
def adjusted_rand_index(group_dict, Ylist, k):
    '''
    INPUT:
    group_dict - (dict) 类别字典
    Ylist - (list) 类别标签列表
    k - (int) 设定的类别数

    OUTPUT:
    (int) 调整兰德系数

    '''
    group_array = np.zeros((k, k))  # 定义一个数组，用来保存聚类所产生的类别标签与给定的外部标签各类别之间共同包含的数据数量
    y_dict = {}  # 定义一个空字典，用来保存外部标签中各类所包含的数据，结构与group_dict相同
    for i in range(len(Ylist)):
        if Ylist[i] not in y_dict:
            y_dict[Ylist[i]] = [i]
        else:
            y_dict[Ylist[i]].append(i)
    # 循环计算group_array的值
    for i in range(k):
        for j in range(k):
            for n in range(len(Ylist)):
                if n in group_dict[list(group_dict.keys())[i]] and n in y_dict[list(y_dict.keys())[j]]:
                    group_array[i][j] += 1  # 如果数据n同时在group_dict的类别i和y_dict的类别j中，group_array[i][j]的数值加一
    RI = 0  # 定义兰德系数(RI)
    sum_i = np.zeros(k)  # 定义一个数组，用于保存聚类结果group_dict中每一类的个数
    sum_j = np.zeros(k)  # 定义一个数组，用于保存外部标签y_dict中每一类的个数
    for i in range(k):
        for j in range(k):
            sum_i[i] += group_array[i][j]
            sum_j[j] += group_array[i][j]
            if group_array[i][j] >= 2:
                RI += comb(group_array[i][j], 2)  # comb用于计算group_array[i][j]中两两组合的组合数
    ci = 0  # ci保存聚类结果中同一类中的两两组合数之和
    cj = 0  # cj保存外部标签中同一类中的两两组合数之和
    for i in range(k):
        if sum_i[i] >= 2:
            ci += comb(sum_i[i], 2)
    for j in range(k):
        if sum_j[j] >= 2:
            cj += comb(sum_j[j], 2)
    E_RI = ci * cj / comb(len(Ylist), 2)  # 计算RI的期望
    max_RI = (ci + cj) / 2  # 计算RI的最大值
    return (RI - E_RI) / (max_RI - E_RI)  # 返回调整兰德系数的值


if __name__ == "__main__":
    print("开始测试聚合聚类算法……")
    Xarray, Ylist = load_data('./data/iris.data')  # 加载数据
    start = time.time()  # 保存开始时间
    Xarray = normalize(Xarray)  # 对特征数据进行标准化处理
    k = 3  # 设定聚类数为3
    dists = distances(Xarray)  # 计算特征数据的距离数组
    print(dists)
    group_dict = hierarchical_clustering(Xarray, k, dists)  # 进行层次聚类
    end = time.time()  # 保存结束时间
    print(group_dict)
    ARI = adjusted_rand_index(group_dict, Ylist, k)  # 计算ARI用来评估聚类结果
    print('Adjusted Rand Index:', ARI)
    print('Time:', end - start)

    print("------------------------------------------")
    print("开始测试K-means算法……")
    Xarray, Ylist = load_data('./data/iris.data')  # 加载数据
    start = time.time()  # 保存开始时间
    Xarray = normalize(Xarray)  # 对特征数据进行标准化处理
    k = 3  # 设定聚类数为3
    iters = 2  # 设定迭代次数为2
    group_dict, scores = k_means(Xarray, k, iters)  # 进行k均值聚类
    end = time.time()  # 保存结束时间
    print('Time:', end - start)
    plt.plot(range(iters), scores)  # 绘制ARI得分折线图
    plt.show()

