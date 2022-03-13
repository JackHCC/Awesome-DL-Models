#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :_utils.py
@Author  :JackHCC
@Date    :2022/3/12 22:22 
@Desc    :

'''
from torch.utils.data import DataLoader
import torch
import time
import torchvision
import torchvision.transforms as transforms

mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True,
                                               transform=transforms.ToTensor())


def _load_data(train_data, test_data, batch_size):
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter


def load_data(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True,
                                                   transform=transforms.ToTensor())
    return _load_data(mnist_train, mnist_test, batch_size)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
        return acc_sum / n


def train(net, train_iter, test_iter, batch_size, loss, optimizer, device, num_epoch):
    # 训练网络基本步骤：
    # Step 1. 加载模型到指定的设备中
    net = net.to(device)
    print("Begin Training on: ", device)
    # Step 2. 开始epoch循环计算，并定义存储计算损失的和评估指标，计时等必要的变量
    for epoch in range(num_epoch):
        train_ls_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        # Step 3. 开始从训练数据迭代器取数据计算，并将取出的数据放到指定的计算设备
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            # Step 4. 将优化器的初始梯度置零
            optimizer.zero_grad()
            # Step 5. 将训练特征数据送进模型计算输出
            y_hat = net(X)
            # Step 6. 计算损失函数的损失并进行反向传播
            # CrossEntropyLoss()函数第一个参数必须传入分类类别个数的维度矩阵，第二个参数可以是维度矩阵，也可以是一个slice
            # 这个slice表示维度矩阵哪一维是label，即自动转为对应维度的One-Hot编码。因此loss第一个参数一般传预测结果，第二参数传训练集label
            # 详细看源码torch.nn.CrossEntropyLoss()
            ls = loss(y_hat, y)
            ls.backward()
            # Step 7. 按步骤执行迭代器
            optimizer.step()
            # Step 8. 统计相关变量的值
            train_ls_sum += ls
            # argmax()返回指定维度最大值的序号
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            # n += y.shape[0]
            n += batch_size
            batch_count += 1
        # Step 9. 验证集对本次训练的网络进行评估
        test_acc = evaluate_accuracy(test_iter, net)
        # Step 10. 打印单词epoch输出的信息
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_ls_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def save_model(net):
    if isinstance(net, torch.nn.Module):
        torch.save(net.state_dict(), './model/model.params')
