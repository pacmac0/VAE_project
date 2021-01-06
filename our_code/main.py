#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os
import torch

for p in ["snapshots/freyfaces", "snapshots/mnist", "plots"]:
    if not os.path.exists(p):
        os.makedirs(p)

if torch.cuda.is_available():
    dev = "cuda"
    print("Using GPU Cuda")
else:
    dev = "cpu"
    torch.set_num_threads(len(os.sched_getaffinity(0)))  # threading on cpu only
    print("Using CPU")

device = torch.device(dev)

mnist_config = {
    "dataset_name": "static_mnist",
    "prior": "vamp",
    # "prior": "mog",
    # "prior": "standard",
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
    "file_name_model": "./snapshots/mnist/mnist",
    "pseudo_from_data": True,
    "device": device,
}


frey_config = {
    "dataset_name": "freyfaces",
    "prior": "vamp",
    # "prior": "mog",
    # "prior": "standard",
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "continues",
    "learning_rate": 0.0005,
    "epochs": 5,
    "file_name_model": "./snapshots/freyfaces/freyfaces",
    "pseudo_from_data": True,
    "device": device,
}


# download http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat to "datasets/freyfaces/frey_rawface.mat"

# vampprior
freyfaces(frey_config)
mnist(mnist_config)

frey_config["prior"] = "standard"
mnist_config["prior"] = "standard"
freyfaces(frey_config)
mnist(mnist_config)


frey_config["prior"] = "mog"
mnist_config["prior"] = "mog"
freyfaces(frey_config)
mnist(mnist_config)
