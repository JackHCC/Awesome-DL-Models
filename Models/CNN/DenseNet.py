#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :DenseNet.py
@Author  :JackHCC
@Date    :2022/3/13 18:56 
@Desc    :

'''
import torch
from torch import nn


def conv_block(in_channel, num_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, num_channel, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_conv, in_channel, num_channel):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_conv):
            layer.append(conv_block(num_channel * i + in_channel, num_channel))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


def transition_block(in_channel, num_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, num_channel, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class DenseNet(nn.Module):
    def __init__(self, num_channel, growth_rate, num_conv):
        super(DenseNet, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.num_channel = num_channel
        self.growth_rate = growth_rate
        self.num_conv = num_conv
        self.blks = []

        self.__make_layer()

        self.net = nn.Sequential(
            self.block,
            *self.blks,
            nn.BatchNorm2d(self.num_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_channel, 10)
        )

    def __make_layer(self):
        for i, num_convs in enumerate(self.num_conv):
            self.blks.append(DenseBlock(num_convs, self.num_channel, self.growth_rate))
            # 上一个稠密块的输出通道数
            self.num_channel += num_convs * self.growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(self.num_conv) - 1:
                self.blks.append(transition_block(self.num_channel, self.num_channel // 2))
                self.num_channel = self.num_channel // 2

    def forward(self, x):
        return self.net(x)


# Test DenseBlock
# blk = DenseBlock(2, 3, 10)
# X = torch.randn(4, 3, 8, 8)
# Y = blk(X)
# print(Y.shape)

# Test DenseNet
# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_conv = [4, 4, 4, 4]
model = DenseNet(num_channels, growth_rate, num_conv)

X = torch.randn(4, 1, 224, 224)
Y = model(X)
print(Y.shape)
