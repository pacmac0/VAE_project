#!/usr/bin/env python3
from scipy.io import loadmat
import time
import torch
import numpy as np
import torch.optim as optim
from VAE import VAE
from eval_generate import generate


def train(model, train_loader, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        start_epoch_time = time.time()

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
            loss, RE, KL = model.get_loss(inputs, mean_dec, z, mean_enc, logvar_enc)
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

        with open(config["model_path"], "wb") as f:
            torch.save(model, f)


def test(model, test_loader):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for inputs in test_loader:
        # get input, data as the list of [inputs, label]
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

    train(
        model,
        train_loader,
        config,
    )

    test(model, val_loader)
    generate(config["model_path"], config["input_size"])
