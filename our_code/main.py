#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os
import torch

for p in ['snapshots/freyfaces', 'snapshots/mnist', "plots"]:
    if not os.path.exists(p):
        os.makedirs(p)

if torch.cuda.is_available():
    dev = 'cuda'
    print("Using GPU Cuda")
else:
    dev = 'cpu'
    torch.set_num_threads(8) # threading on cpu only
    print("Using CPU")

device = torch.device(dev)

mnist_config = {
    "dataset_name": "static_mnist",
    "prior": "standard", # "vamp", "mog"
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "test_batch_size": 100,
    "input_size": [1, 28, 28],
    "input_type": "binary",
    "pseudoinputs_std": 0.01,
    "pseudoinputs_mean": 0.05,
    "learning_rate": 0.0005,
    "epochs": 5,
    "file_name_model": "./snapshots/mnist/final_mnist.model",
    'pseudo_from_data': True,
    "device": device
}


frey_config = {
    "dataset_name": "freyfaces",
    "prior": "standard",  # "vamp", # standard
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "binary",
    "learning_rate": 0.0005,
    "epochs": 5,
    "file_name_model": "./snapshots/freyfaces/final_freyfaces.model",
    "pseudo_from_data": True,
    "device": device
}


# download http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat to "datasets/freyfaces/frey_rawface.mat"
freyfaces(frey_config)

mnist(mnist_config)
