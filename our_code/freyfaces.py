#!/usr/bin/env python3
from scipy.io import loadmat
import time
import torch
import numpy as np
import torch.optim as optim
from VAE import VAE, train, test
from eval_generate import generate


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
    val_loader = torch.utils.data.DataLoader(
        ff_torch[train_size:], config["batch_size"], shuffle=True
    )

    model = VAE(config)
    model.to(config["device"])

    train(
        model,
        train_loader,
        config,
    )

    test(model, val_loader, config)
    generate(config["file_name_model"], config["input_size"], "freyfaces.png")
