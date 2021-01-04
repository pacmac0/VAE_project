import datetime
import os.path as osp
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import torch.utils.data as data_utils
from VAE import VAE
from eval_generate import generate

if not os.path.exists('models'):
    os.makedirs('models')

config = {
    "prior": "vamp",  # standard
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "batch_size": 100,
    "test_batch_size": 100,
    "input_size": [1, 28, 28],
    "input_type": "binary",
    "learning_rate": 0.0005,
    "epochs": 2000,
    "pseudo_from_data": True,
    "model_path": "./models/mnist"
}

# use GPU
if torch.cuda.is_available():
    dev = "cuda"
    print("Cuda")
else:
    dev = "cpu"
    torch.set_num_threads(8)  # threading on cpu only
    print("CPU")

device = torch.device(dev)


def load_static_mnist(args):
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
        batch_size=args["batch_size"],
        shuffle=True,
    )

    eval_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(evaluation_data), torch.from_numpy(evaluation_labels)
        ),
        batch_size=args["test_batch_size"],
        shuffle=False,
    )

    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(test_data), torch.from_numpy(test_labels)
        ),
        batch_size=args["test_batch_size"],
        shuffle=True,
    )

    # get pseudo init params from random data
    if args["pseudo_from_data"] == True:
        args["pseudo_std"] = 0.01
        np.random.shuffle(train_data)
        # print("DIM: {}".format(train_data.shape))
        dat = train_data[
            0 : int(args["pseudo_components"])
        ].T  # make columns components(data-points)
        # print("DIM: {}".format(dat.shape))
        rand_std_norm = np.random.randn(
            np.prod(args["input_size"]), args["pseudo_components"]
        )
        args["pseudo_mean"] = torch.from_numpy(
            dat + args["pseudo_std"] * rand_std_norm
        ).float()

    return train_loader, eval_loader, test_loader, args



def training(model, train_loader, epochs, warmup_period, learning_rate=0.0005):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / warmup_period
        if beta > 1.0:
            beta = 1.0
        print(f"--> beta: {beta}")

        train_loss = []
        train_re = []
        train_kl = []

        start_epoch_time = time.time()

        for i, (inputs, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # print("\nTraining batch #", i)

            inputs = inputs.to(device)

            # forward
            mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(inputs)

            # calculate loss
            loss, RE, KL = model.get_loss(
                inputs, mean_dec, z, mean_enc, logvar_enc, beta=beta
            )

            # backpropagate
            loss.backward()

            if i == len(train_loader) / 2:
                print("loss", loss.item(), "RE", RE.item(), "KL", KL.item())

            optimizer.step()

            # collect epoch statistics
            train_loss.append(loss.item())
            train_re.append(RE.item())
            train_kl.append(KL.item())

        epoch_loss = sum(train_loss) / len(train_loader)
        epoch_re = sum(train_re) / len(train_loader)
        epoch_kl = sum(train_kl) / len(train_loader)

        end_epoch_time = time.time()
        epoch_time_diff = end_epoch_time - start_epoch_time

        print(
            f"Epoch: {epoch}; loss: {epoch_loss}, RE: {epoch_re}, KL: {epoch_kl}, time elapsed: {epoch_time_diff}"
        )

        # save parameters
        with open("./snapshots/mnist_model", "wb") as f:
            torch.save(model, f)


def testing(model, train_loader, test_loader):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for (inputs, _) in test_loader:
        # get input, data as the list of [inputs, label]
        inputs = inputs.to(device)

        mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(inputs)
        loss, RE, KL = model.get_loss(
            inputs, mean_dec, z, mean_enc, logvar_enc, beta=1.0
        )

        test_loss.append(loss.item())
        test_re.append(RE.item())
        test_kl.append(KL.item())

    mean_loss = sum(test_loss) / len(test_loader)
    mean_re = sum(test_re) / len(test_loader)
    mean_kl = sum(test_kl) / len(test_loader)

    print(f"Test results: loss avg: {mean_loss}, RE avg: {mean_re}, KL: {mean_kl}")


if __name__ == "__main__":
    torch.manual_seed(14)
    # TODO: refactor load_static_mnist
    train_loader, eval_loader, test_loader, args = load_static_mnist(config)

    model = VAE(args)
    model.to(device)

    print("--> Starting training")
    start_time = time.time()

    epochs = args["epochs"]
    warmup = args["warmup"]
    learning_rate = args["learning_rate"]
    training(model, train_loader, epochs, warmup, learning_rate)
    end_time = time.time()
    time_diff = end_time - start_time
    print("Training done, time elapsed: ", time_diff)
    testing(model, train_loader, test_loader)
    generate(config["model_path"])
