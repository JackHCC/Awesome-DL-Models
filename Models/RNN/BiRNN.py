#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-DL-Models 
@File    :BiRNN.py
@Author  :JackHCC
@Date    :2022/3/14 19:17 
@Desc    :

'''
import torch
from torch import nn

from _utils import load_data_time_machine, train
from RNN import RNNModel


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_layer, device = len(vocab), 256, 2, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs, lr = 500, 1

    num_inputs = vocab_size

    """
    注意API内的可以通过设置 bidirectional=True 构建双向RNN
    """
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layer, bidirectional=True)

    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)

    train(model, train_iter, vocab, lr, num_epochs, device)