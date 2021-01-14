#!/usr/bin/env python3
import torch
import os

if torch.cuda.is_available():
    dev = "cuda"
    print("Using GPU Cuda")
else:
    dev = "cpu"
    torch.set_num_threads(os.cpu_count())  # threading on cpu only
    print("Using CPU")

device = torch.device(dev)
mnist_vamp = {
    "dataset_name": "mnist",
    "prior": "vamp",
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
    "epochs": 1000,
    "pseudo_from_data": True,
    "device": device,
}

mnist_standard = {
    "dataset_name": "mnist",
    "prior": "standard",
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
    "epochs": 1000,
    "pseudo_from_data": False,
    "device": device,
}


mnist_mog = {
    "dataset_name": "mnist",
    "prior": "mog",
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
    "epochs": 1000,
    "pseudo_from_data": False,
    "device": device,
}


frey_vamp = {
    "dataset_name": "freyfaces",
    "prior": "vamp",
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "cont",
    "learning_rate": 0.0005,
    "epochs": 1000,
    "pseudo_from_data": True,
    "device": device,
}

frey_mog = {
    "dataset_name": "freyfaces",
    "prior": "mog",
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "cont",
    "learning_rate": 0.0005,
    "epochs": 1000,
    "pseudo_from_data": False,
    "device": device,
}


frey_standard = {
    "dataset_name": "freyfaces",
    "prior": "standard",
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "cont",
    "learning_rate": 0.0005,
    "epochs": 1000,
    "pseudo_from_data": False,
    "device": device,
}
