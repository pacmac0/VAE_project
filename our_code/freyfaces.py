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

    # get pseudo init params from random train_data
    # and add some randomness to it is not the exactly the same
    if config["pseudo_from_data"] and config["prior"] == "vamp":
        config["pseudo_std"] = 0.01
        np.random.shuffle(train_data)
        # print("DIM: {}".format(train_data.shape))
        dat = train_data[
            0 : int(config["pseudo_components"])
        ].T  # make columns components(train_data-points)
        # print("DIM: {}".format(dat.shape))
        # add some randomness to the pseudo inputs to avoid overfitting
        rand_std_norm = np.random.randn(
            np.prod(config["input_size"]), config["pseudo_components"]
        )
        config["pseudo_mean"] = torch.from_numpy(
            dat + config["pseudo_std"] * rand_std_norm
        ).float()
    else:
        config["pseudo_std"] = 0.01
        config["pseudo_mean"] = 0.05
    model = VAE(config)
    model.to(config["device"])

    train(
        model,
        train_loader,
        config,
    )

    test(model, val_loader, config)
    generate(config["file_name_model"], config["input_size"], "freyfaces.png")
