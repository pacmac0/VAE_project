#!/usr/bin/env python3
from scipy.io import loadmat
import time
import torch
import numpy as np
import torch.optim as optim
from VAE import VAE, train, test, add_pseudo_prior
from eval_generate import generate


def freyfaces(config):
    path = "datasets/freyfaces/frey_rawface.mat"

    train_data = loadmat(path)
    # static mnist shape: torch.Size([100, 784])
    train_data = train_data["ff"].T.reshape((-1, 28 * 20)).astype("float32") / 255.0
    train_data = train_data[: int(len(train_data) / config["batch_size"]) * config["batch_size"]]
    np.random.shuffle(train_data)
    ff_torch = torch.from_numpy(train_data)

    train_size = 1765
    train_loader = torch.utils.data.DataLoader(
        ff_torch[:train_size], config["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        ff_torch[train_size:], config["batch_size"], shuffle=True
    )

    add_pseudo_prior(config, train_data)
    model = VAE(config)
    model.to(config["device"])

    train(
        model,
        train_loader,
        config,
    )

    test(model, val_loader, config)
    generate(config["file_name_model"], config["input_size"], "freyfaces.png")
