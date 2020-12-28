#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def xavier_init(m):
    s =  np.sqrt( 2. / (m.linear.in_features + m.linear.out_features) )
    m.weight.data.normal_(0, s)


def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

def init_layer_weights(layer):
    torch.nn.init.xavier_uniform(layer.weight)
