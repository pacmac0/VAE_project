import torch
import os
import numpy as np
import time

import torch.utils.data as data_utils
from VAE import VAE, train, test, add_pseudo_prior
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

    add_pseudo_prior(config, train_data)
    return train_loader, eval_loader, test_loader


def mnist(config):
    print(config)
    torch.manual_seed(14)

    train_loader, eval_loader, test_loader = load_static_mnist(config)

    # If a snapshot exist in /snapshots then use trained weights
    model = VAE(config)
    model.to(config["device"])

    print("Starting train")
    start_time = time.time()

    train(
        model,
        train_loader,
        config,
        test_loader
    )
    end_time = time.time()
    time_diff = end_time - start_time
    print("Training done, time elapsed: ", time_diff)
    print("Testing on test data")
    test(model, test_loader, config)
