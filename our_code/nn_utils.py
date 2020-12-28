#!/usr/bin/env python3
import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_layer_weights(layer):
    torch.nn.init.xavier_uniform(layer.weight)
