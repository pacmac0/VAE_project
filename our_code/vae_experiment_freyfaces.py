#!/usr/bin/env python3
from scipy.io import loadmat
import torch
import numpy as np
from VAE import VAE, training, testing

config = {
    #"seed": 14,
    #"dataset_name": "static_mnist",
    #"model_name": "vae",
    "prior": "standard", #"vamp", # standard
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
    #"pseudoinputs_std": 0.01,
    #"pseudoinputs_mean": 0.05,
    "learning_rate": 0.0005,
    "max_epoch": 2000,
    "file_name_model": "./snapshots/model.model",
}

if torch.cuda.is_available():
    dev = 'cuda'
    print("--> Using GPU Cuda")
else:
    dev = 'cpu'
    torch.set_num_threads(8) # threading on cpu only
    print("--> Using CPU")

device = torch.device(dev)

# DOWNLOAD FROM HERE: http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat
if __name__ == "__main__":
    path = "datasets/freyfaces/frey_rawface.mat"
    batch_size = 64

    ff = loadmat(path)
    ff = ff["ff"].T.reshape((-1, 1, 28, 20)).astype('float32')/255.
    ff = ff[:int(len(ff)/batch_size)*batch_size]
    np.random.shuffle(ff)
    ff_torch = torch.from_numpy(ff)

    train_size = 1765
    train = ff_torch[:train_size]
    val = ff_torch[train_size:]

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)

    model = VAE(config)
    model.to(device)

    training(model, train_loader, config["max_epoch"], config["warmup"], config["file_name_model"],
             config["learning_rate"])

    testing(model, None, val_loader)
