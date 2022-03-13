#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :GoogleNet.py
@Author  :JackHCC
@Date    :2022/3/13 14:26 
@Desc    :

'''
import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channel, c1_channel, c2_channel, c3_channel, c4_channel):
        super(Inception, self).__init__()
        # Path1: 1 * 1 Conv
        self.path1 = nn.Conv2d(in_channels=in_channel, out_channels=c1_channel, kernel_size=1)
        # Path2: 1 * 1 Conv + 3 * 3 Conv
        self.path2_1 = nn.Conv2d(in_channels=in_channel, out_channels=c2_channel[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(in_channels=c2_channel[0], out_channels=c2_channel[1], kernel_size=3, padding=1)
        # Path3: 1 * 1 Conv + 5 * 5 Conv
        self.path3_1 = nn.Conv2d(in_channels=in_channel, out_channels=c3_channel[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(in_channels=c3_channel[0], out_channels=c3_channel[1], kernel_size=5, padding=2)
        # Path4: 3 * 3 MaxPool + 1 * 1 Conv
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channels=in_channel, out_channels=c4_channel, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        p1 = self.relu(self.path1(x))
        p2 = self.relu(self.path2_2(self.relu(self.path2_1(x))))
        p3 = self.relu(self.path3_2(self.relu(self.path3_1(x))))
        p4 = self.relu(self.path4_2(self.path4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x


input1 = torch.rand(size=(1, 1, 96, 96))
model = GoogleNet()
# print(list(model.parameters()))
print(model.state_dict())
output1 = model(input1)
print(output1.shape)
