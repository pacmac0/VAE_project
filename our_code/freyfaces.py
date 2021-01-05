#!/usr/bin/env python3
from scipy.io import loadmat
import time
import torch
import numpy as np
import torch.optim as optim
from VAE import VAE, training, testing
from eval_generate import generate


if torch.cuda.is_available():
    dev = "cuda"
    print("GPU")
else:
    dev = "cpu"
    torch.set_num_threads(8)  # threading on cpu only
    print("CPU")

device = torch.device(dev)

def freyfaces(config):
    path = "datasets/freyfaces/frey_rawface.mat"

    data = loadmat(path)
    # static mnist shape: torch.Size([100, 784])
    data = data["ff"].T.reshape((-1, 28 * 20)).astype("float32") / 255.0
    data = data[: int(len(data) / config["batch_size"]) * config["batch_size"]]
    np.random.shuffle(data)
    ff_torch = torch.from_numpy(data)

    train_size = 1765
    train_loader = torch.utils.data.DataLoader(
        ff_torch[:train_size], config["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(ff_torch[train_size:], config["batch_size"], shuffle=True)

    model = VAE(config)
    model.to(device)

    training(
        model,
        train_loader,
        config,
    )

    testing(model, val_loader, config)
    generate(config["file_name_model"], config["input_size"])
