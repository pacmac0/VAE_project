#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os
import torch

for experiment in ["freyfaces", "mnist"]:
    for prior in ["mog", "vamp", "standard"]:
        for subfolder in ["images", "models"]:
            p = f'experiments/{experiment}/{prior}/{subfolder}'
            if not os.path.exists(p):
                os.makedirs(p)


if torch.cuda.is_available():
    dev = "cuda"
    print("Using GPU Cuda")
else:
    dev = "cpu"
    #torch.set_num_threads(len(os.sched_getaffinity(0)))  # threading on cpu only
    torch.set_num_threads(8)  # threading on cpu only
    print("Using CPU")

device = torch.device(dev)

mnist_config = {
    "dataset_name": "mnist",
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
    "epochs": 10,
    "pseudo_from_data": False,
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
    "input_type": "cont",
    "learning_rate": 0.0005,
    "epochs": 10,
    "pseudo_from_data": False,
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
