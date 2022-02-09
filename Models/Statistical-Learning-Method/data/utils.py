#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :utils.py
@Author  :JackHCC
@Date    :2022/2/8 12:16 
@Desc    :

'''
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords


def load_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径

    OUTPUT:
    Xarray - (array) 特征数据数组
    Ylist - (list) 类别标签列表

    '''
    Xlist = []  # 定义一个列表用来保存每条数据
    Ylist = []  # 定义一个列表用来保存每条数据的类别标签
    fr = open(file)
    for line in fr.readlines():  # 逐行读取数据，鸢尾花数据集每一行表示一个鸢尾花的特征和类别标签，用逗号分隔
        cur = line.split(',')
        label = cur[-1]
        X = [float(x) for x in cur[:-1]]  # 用列表来表示一条特征数据
        Xlist.append(X)
        Ylist.append(label)
    Xarray = np.array(Xlist)  # 将特征数据转换为数组类型，方便之后的操作
    print('Data shape:', Xarray.shape)
    print('Length of labels:', len(Ylist))
    return Xarray, Ylist


"""
！！！注意需要下载ntlk语料库，如果报错可以去：
https://github.com/nltk/nltk_data
下载数据集，放在报错提示的任意一个文件夹下！！！
"""
def load_lsa_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径

    OUTPUT:
    org_topics - (list) 原始话题标签列表
    text - (list) 文本列表
    words - (list) 单词列表

    '''
    df = pd.read_csv(file)  # 读取文件
    org_topics = df['category'].unique().tolist()  # 保存文本原始的话题标签
    df.drop('category', axis=1, inplace=True)
    n = df.shape[0]  # n为文本数量
    text = []
    words = []
    for i in df['text'].values:
        t = i.translate(str.maketrans('', '', string.punctuation))  # 去除文本中的标点符号
        t = [j for j in t.split() if j not in stopwords.words('english')]  # 去除文本中的停止词
        t = [j for j in t if len(j) > 3]  # 长度小于等于3的单词大多是无意义的，直接去除
        text.append(t)  # 将处理后的文本保存到文本列表中
        words.extend(set(t))  # 将文本中所包含的单词保存到单词列表中
    words = list(set(words))  # 去除单词列表中的重复单词
    return org_topics, text, words
