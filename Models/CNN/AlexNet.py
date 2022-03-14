#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :AlexNet.py
@Author  :JackHCC
@Date    :2022/3/12 22:20 
@Desc    :

'''
import torch
import torch.nn as nn

"""
CNN:
    out_size = floor((input_size  + padding * 2 - kernel_size) / stride + 1)
Pool:
    out_size = floor((input_size + padding * 2 - kernel_size) / stride + 1)
"""

# 输入224*224
AlexNet = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),  # 96 * 54 * 54
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),  # 96 * 26 * 26
    nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 256 * 26 * 26
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),  # 256 * 12 * 12
    nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 384 * 12 * 12
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 384 * 12 * 12
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256 * 12 * 12
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),  # 256 * 5 * 5
    nn.Flatten(),
    nn.Linear(256 * 5 * 5, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)


# Test Net
X = torch.randn(1, 1, 224, 224)
for layer in AlexNet:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
