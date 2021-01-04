#!/usr/bin/env python3
from scipy.io import loadmat
import time
import torch
import numpy as np
import torch.optim as optim
from VAE import VAE


def training(
    model, train_loader, max_epoch, warmup_period, file_name, learning_rate=0.0005
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, max_epoch + 1):
        start_epoch_time = time.time()
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / warmup_period
        if beta > 1.0:
            beta = 1.0

        train_loss = []
        train_re = []
        train_kl = []

        for inputs in train_loader:
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


def testing(model, test_loader):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for i, data in enumerate(test_loader):
        # get input, data as the list of [inputs, label]
        inputs = data
        inputs = inputs.to(device)

        mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(inputs)
        loss, RE, KL = model.get_loss(inputs, mean_dec, z, mean_enc, logvar_enc)

        test_loss.append(loss.item())
        test_re.append(RE.item())
        test_kl.append(KL.item())

    mean_loss = sum(test_loss) / len(test_loader)
    mean_re = sum(test_re) / len(test_loader)
    mean_kl = sum(test_kl) / len(test_loader)

    print(f"Test results: loss avg: {mean_loss}, RE avg: {mean_re}, KL: {mean_kl}")


config = {
    # "seed": 14,
    # "dataset_name": "static_mnist",
    # "model_name": "vae",
    "prior": "standard",  # "vamp", # standard
    "pseudo_components": 500,
    "warmup": 100,
    "z1_size": 40,
    # "z2_size": 40,
    "batch_size": 100,
    "input_size": [1, 28, 20],
    "input_type": "binary",
    # "dynamic_binarization": False,
    # "use_training_data_init": 1,
    # "pseudoinputs_std": 0.01,
    # "pseudoinputs_mean": 0.05,
    "learning_rate": 0.0005,
    "max_epoch": 2000,
    "file_name_model": "./snapshots/model.model",
}

if torch.cuda.is_available():
    dev = "cuda"
    print("--> Using GPU Cuda")
else:
    dev = "cpu"
    torch.set_num_threads(8)  # threading on cpu only
    print("--> Using CPU")

device = torch.device(dev)

# DOWNLOAD FROM HERE: http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat
if __name__ == "__main__":
    path = "datasets/freyfaces/frey_rawface.mat"

    ff = loadmat(path)
    # static mnist shape: torch.Size([100, 784])
    ff = ff["ff"].T.reshape((-1, 28 * 20)).astype("float32") / 255.0
    ff = ff[: int(len(ff) / config["batch_size"]) * config["batch_size"]]
    np.random.shuffle(ff)
    ff_torch = torch.from_numpy(ff)

    train_size = 1765
    train = ff_torch[:train_size]
    val = ff_torch[train_size:]

    train_loader = torch.utils.data.DataLoader(
        train, config["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val, config["batch_size"], shuffle=True)

    model = VAE(config)
    model.to(device)

    training(
        model,
        train_loader,
        config["max_epoch"],
        config["warmup"],
        config["file_name_model"],
        config["learning_rate"],
    )

    testing(model, val_loader)
