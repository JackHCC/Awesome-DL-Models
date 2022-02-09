#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :latent_semantic_analysis.py
@Author  :JackHCC
@Date    :2022/2/9 15:27
@Desc    :Implement LSA

'''
import numpy as np
import time
from data.utils import load_lsa_data


# 定义构建单词-文本矩阵的函数，这里矩阵的每一项表示单词在文本中的出现频次，也可以用TF-IDF来表示
def frequency_counter(text, words):
    '''
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表

    OUTPUT:
    X - (array) 单词-文本矩阵

    '''
    X = np.zeros((len(words), len(text)))  # 定义m*n的矩阵，其中m为单词列表中的单词个数，n为文本个数
    for i in range(len(text)):
        t = text[i]  # 读取文本列表中的第i条文本
        for w in t:
            ind = words.index(w)  # 取出第i条文本中的第t个单词在单词列表中的索引
            X[ind][i] += 1  # 对应位置的单词出现频次加一
    return X


# 定义潜在语义分析函数
def do_lsa(X, k, words):
    '''
    INPUT:
    X - (array) 单词-文本矩阵
    k - (int) 设定的话题数
    words - (list) 单词列表

    OUTPUT:
    topics - (list) 生成的话题列表

    '''
    w, v = np.linalg.eig(np.matmul(X.T, X))  # 计算Sx的特征值和特征向量，其中Sx=X.T*X，Sx的特征值w即为X的奇异值分解的奇异值，v即为对应的奇异向量
    sort_inds = np.argsort(w)[::-1]  # 对特征值降序排列后取出对应的索引值
    w = np.sort(w)[::-1]  # 对特征值降序排列
    V_T = []  # 用来保存矩阵V的转置
    for ind in sort_inds:
        V_T.append(v[ind] / np.linalg.norm(v[ind]))  # 将降序排列后各特征值对应的特征向量单位化后保存到V_T中
    V_T = np.array(V_T)  # 将V_T转换为数组，方便之后的操作
    Sigma = np.diag(np.sqrt(w))  # 将特征值数组w转换为对角矩阵，即得到SVD分解中的Sigma
    U = np.zeros((len(words), k))  # 用来保存SVD分解中的矩阵U
    for i in range(k):
        ui = np.matmul(X, V_T.T[:, i]) / Sigma[i][i]  # 计算矩阵U的第i个列向量
        U[:, i] = ui  # 保存到矩阵U中
    topics = []  # 用来保存k个话题
    for i in range(k):
        inds = np.argsort(U[:, i])[
               ::-1]  # U的每个列向量表示一个话题向量，话题向量的长度为m，其中每个值占向量值之和的比重表示对应单词在当前话题中所占的比重，这里对第i个话题向量的值降序排列后取出对应的索引值
        topic = []  # 用来保存第i个话题
        for j in range(10):
            topic.append(words[inds[j]])  # 根据索引inds取出当前话题中比重最大的10个单词作为第i个话题
        topics.append(' '.join(topic))  # 保存话题i
    return topics


if __name__ == "__main__":
    print("开始测试LSA……")
    org_topics, text, words = load_lsa_data('./data/bbc_text.csv')  # 加载数据
    print('Original Topics:')
    print(org_topics)  # 打印原始的话题标签列表
    start = time.time()  # 保存开始时间
    X = frequency_counter(text, words)  # 构建单词-文本矩阵
    k = 5  # 设定话题数为5
    topics = do_lsa(X, k, words)  # 进行潜在语义分析
    print('Generated Topics:')
    for i in range(k):
        print('Topic {}: {}'.format(i + 1, topics[i]))  # 打印分析后得到的每个话题
    end = time.time()  # 保存结束时间
    print('Time:', end - start)
