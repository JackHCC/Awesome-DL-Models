#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :LeNet.py
@Author  :JackHCC
@Date    :2022/3/12 19:39 
@Desc    :Implement LeNet-5 and summarize the basic process of model training

'''
import torch
import torch.nn as nn
from _utils import load_data, train, save_model


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 注意: 全连接层的输入输出feature是需要根据输入输出计算的，这里最初输入图片式28*28
        self.fc1 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        # 将一个矩阵展平，第一维不动：可以使用view或flatten方法
        x = x.view(x.size()[0], -1)     # 注意获取一个tensor的尺寸方法：x.size()
        # x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


"""
    注意：下面这些通用代码后续移到_utils.py
"""
# def load_data(train_data, test_data, batch_size):
#     train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#     return train_iter, test_iter
#
#
# def evaluate_accuracy(data_iter, net, device=None):
#     if device is None and isinstance(net, torch.nn.Module):
#         # 如果没指定device就使用net的device
#         device = list(net.parameters())[0].device
#     acc_sum, n = 0.0, 0
#     with torch.no_grad():
#         for X, y in data_iter:
#             net.eval()  # 评估模式, 这会关闭dropout
#             acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
#             net.train()  # 改回训练模式
#             n += y.shape[0]
#         return acc_sum / n
#
#
# def train(net, train_iter, test_iter, batch_size, loss, optimizer, device, num_epoch):
#     # 训练网络基本步骤：
#     # Step 1. 加载模型到指定的设备中
#     net = net.to(device)
#     print("Begin Training on: ", device)
#     # Step 2. 开始epoch循环计算，并定义存储计算损失的和评估指标，计时等必要的变量
#     for epoch in range(num_epoch):
#         train_ls_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
#         # Step 3. 开始从训练数据迭代器取数据计算，并将取出的数据放到指定的计算设备
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             # Step 4. 将优化器的初始梯度置零
#             optimizer.zero_grad()
#             # Step 5. 将训练特征数据送进模型计算输出
#             y_hat = net(X)
#             # Step 6. 计算损失函数的损失并进行反向传播
#             # CrossEntropyLoss()函数第一个参数必须传入分类类别个数的维度矩阵，第二个参数可以是维度矩阵，也可以是一个slice
#             # 这个slice表示维度矩阵哪一维是label，即自动转为对应维度的One-Hot编码。因此loss第一个参数一般传预测结果，第二参数传训练集label
#             # 详细看源码torch.nn.CrossEntropyLoss()
#             ls = loss(y_hat, y)
#             ls.backward()
#             # Step 7. 按步骤执行迭代器
#             optimizer.step()
#             # Step 8. 统计相关变量的值
#             train_ls_sum += ls
#             # argmax()返回指定维度最大值的序号
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             # n += y.shape[0]
#             n += batch_size
#             batch_count += 1
#         # Step 9. 验证集对本次训练的网络进行评估
#         test_acc = evaluate_accuracy(test_iter, net)
#         # Step 10. 打印单词epoch输出的信息
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#               % (epoch + 1, train_ls_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == "__main__":
    # Step 1: Test LeNet struction，一般第一步写好模型结构，并对模型结果进行测试
    # 一般在写好模型的结构后，需要现生成一些随机数据验证模型维度是否正确
    input1 = torch.rand([5, 1, 28, 28])
    model = LeNet5()
    # 查看模型的参数，主要包括模型的具体参数和requires_grad参数
    # print(list(model.parameters()))
    output1 = model(input1)
    print(output1.shape)

    # Step 2: Load DataSet，数据处理一般包括数据的预处理，借助DataLoader函数生成训练测试数据的迭代器
    # mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True,
    #                                                 transform=transforms.ToTensor())
    # mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True,
    #                                                transform=transforms.ToTensor())

    batch_size = 256
    train_iter, test_iter = load_data(batch_size)

    # Step 3: Parameter Set
    # 一般设置device都采用这种方式可以根据实际情况选择GPU或者CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 80

    # Step 4: Training
    train(model, train_iter, test_iter, batch_size, loss, optimizer, device, num_epochs)

    # Step 5: Save Model and show
    save_model(model)
