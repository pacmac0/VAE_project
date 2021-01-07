#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os
import torch
import configs

for experiment in ["freyfaces", "mnist"]:
    for prior in ["mog", "vamp", "standard"]:
        for subfolder in ["images", "models"]:
            p = f'experiments/{experiment}/{prior}/{subfolder}'
            if not os.path.exists(p):
                os.makedirs(p)




# download http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat to "datasets/freyfaces/frey_rawface.mat"

freyfaces(configs.frey_vamp)
mnist(configs.mnist_vamp)


freyfaces(configs.frey_standard)
mnist(configs.mnist_standard)

freyfaces(configs.frey_mog)
mnist(configs.mnist_mog)
