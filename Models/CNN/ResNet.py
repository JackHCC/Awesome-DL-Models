#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :ResNet.py
@Author  :JackHCC
@Date    :2022/3/13 15:56 
@Desc    :

'''
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1_conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                               stride=strides) if use_1x1_conv else None
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1_conv=True, strides=1))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            *resnet_block(64, 64, 2, first_block=True)
        )
        self.block3 = nn.Sequential(
            *resnet_block(64, 128, 2)
        )
        self.block4 = nn.Sequential(
            *resnet_block(128, 256, 2)
        )
        self.block5 = nn.Sequential(
            *resnet_block(256, 512, 2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


# Test Residual
# blk = Residual(3, 3)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)

# Test ResNet
model = ResNet()
print(model)
X = torch.rand(size=(1, 1, 224, 224))

Y = model(X)
print(Y.shape)
