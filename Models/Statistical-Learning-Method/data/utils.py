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
