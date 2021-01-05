import datetime
import os.path as osp
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import json

import torch.utils.data as data_utils
from VAE import VAE, train, test
from eval_generate import generate


def load_static_mnist(config):
    # load each file separate
    with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_train.amat")
    ) as f:
        lines = f.readlines()
    train_data = np.array([[int(i) for i in l.split()] for l in lines]).astype(
        "float32"
    )
    with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_valid.amat")
    ) as f:
        lines = f.readlines()
    evaluation_data = np.array([[int(i) for i in l.split()] for l in lines]).astype(
        "float32"
    )
    with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_test.amat")
    ) as f:
        lines = f.readlines()
    test_data = np.array([[int(i) for i in l.split()] for l in lines]).astype("float32")

    train_labels = np.zeros((train_data.shape[0], 1))
    evaluation_labels = np.zeros((evaluation_data.shape[0], 1))
    test_labels = np.zeros((test_data.shape[0], 1))

    # pytorch data loader to create tensors
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(train_data), torch.from_numpy(train_labels)
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    eval_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(evaluation_data), torch.from_numpy(evaluation_labels)
        ),
        batch_size=config["test_batch_size"],
        shuffle=False,
    )

    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(test_data), torch.from_numpy(test_labels)
        ),
        batch_size=config["test_batch_size"],
        shuffle=True,
    )

    # get pseudo init params from random data
    # and add some randomness to it is not the exactly the same
    if config["pseudo_from_data"] and config["prior"] == "vamp":
        config["pseudo_std"] = 0.01
        np.random.shuffle(train_data)
        # print("DIM: {}".format(train_data.shape))
        dat = train_data[
            0 : int(config["pseudo_components"])
        ].T  # make columns components(data-points)
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
    return train_loader, eval_loader, test_loader


def mnist(config):
    torch.manual_seed(14)

    train_loader, eval_loader, test_loader = load_static_mnist(config)

    # If a snapshot exist in /snapshots then use trained weights
    file_name = config["file_name_model"]
    model = VAE(config)
    model.to(config["device"])

    print("Starting train")
    start_time = time.time()

    train(
        model,
        train_loader,
        config,
    )
    end_time = time.time()
    time_diff = end_time - start_time
    print("Training done, time elapsed: ", time_diff)
    print("Testing on test data")
    test(model, test_loader, config)
    generate(config["file_name_model"], config["input_size"])
