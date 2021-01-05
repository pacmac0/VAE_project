#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os

for p in ['snapshots/freyfaces', 'snapshots/mnist']:
    if not os.path.exists(p):
        os.makedirs(p)

mnist_config = {
    #"seed": 14,
    "dataset_name": "static_mnist",
    #"model_name": "vae",
    "prior": "standard", # "vamp", "mog"
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    #"z2_size": 40,
    "batch_size": 100,
    "test_batch_size": 100,
    "input_size": [1, 28, 28],
    "input_type": "binary",
    #"dynamic_binarization": False,
    #"use_training_data_init": 1,
    "pseudoinputs_std": 0.01,
    "pseudoinputs_mean": 0.05,
    "learning_rate": 0.0005,
    "max_epoch": 10,
    "file_name_model": "./snapshots/mnist/final_mnist.model",
    'pseudo_from_data': True,
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
    "epochs": 100,
    "model_path": "./snapshots/freyfaces/final_freyfaces.model"
}

# download http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat to "datasets/freyfaces/frey_rawface.mat"
freyfaces(frey_config)

# mnist(mnist_config)
