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
from VAE import VAE
from eval_generate import generate


# use GPU
if torch.cuda.is_available():
    dev = 'cuda'
    print("--> Using GPU Cuda")
else:
    dev = 'cpu'
    torch.set_num_threads(8) # threading on cpu only
    print("--> Using CPU")

device = torch.device(dev) 

def load_static_mnist(config):
    # load each file separate
    with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_train.amat")
    ) as f:
        lines = f.readlines()
    train_data = np.array([[int(i) for i in l.split()] for l in lines]).astype("float32")
    with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_valid.amat")
    ) as f:
        lines = f.readlines()
    evaluation_data = np.array([[int(i) for i in l.split()] for l in lines]).astype("float32")
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
        data_utils.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels)), 
        batch_size=config["batch_size"], shuffle=True)

    eval_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.from_numpy(evaluation_data), torch.from_numpy(evaluation_labels)), 
        batch_size=config["test_batch_size"], shuffle=False)

    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels)), 
        batch_size=config["test_batch_size"], shuffle=True)

    # get pseudo init params from random data
    # and add some randomness to it is not the exactly the same
    if config['pseudo_from_data'] and config['prior'] == 'vamp':
        config['pseudo_std'] = 0.01
        np.random.shuffle(train_data)
        #print("DIM: {}".format(train_data.shape))
        dat = train_data[0 : int(config['pseudo_components']) ].T # make columns components(data-points)
        #print("DIM: {}".format(dat.shape))
        # add some randomness to the pseudo inputs to avoid overfitting
        rand_std_norm = np.random.randn(np.prod(config['input_size']), config['pseudo_components'])
        config['pseudo_mean'] = torch.from_numpy(dat + config['pseudo_std'] * rand_std_norm).float()
    else:
        config['pseudo_std'] = 0.01
        config['pseudo_mean'] = 0.05
    return train_loader, eval_loader, test_loader


def training(model, train_loader, max_epoch, warmup_period, file_name, config,
        learning_rate=0.0005):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    train_re_per_epoch = []
    train_kl_per_epoch = []
    for epoch in range(1, max_epoch+1):
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / warmup_period
        if beta > 1.0:
            beta = 1.0
        print(f"--> beta: {beta}")

        train_loss = []
        train_re = []
        train_kl = []
        train_beta = []

        start_epoch_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # get input, data as the list of [inputs, label]
            inputs, _ = data
            inputs = inputs.to(device)

            # forward
            mean_dec, logvar_dec, z, mean_enc, logvar_enc = \
                model.forward(inputs)

            # calculate loss
            loss, RE, KL = model.get_loss(
                inputs, mean_dec, z, mean_enc, logvar_enc, beta=beta
            )

            # backpropagate
            loss.backward()

            if i == len(train_loader) / 2:
                print(
                    "loss", loss.item(),
                    "RE", RE.item(),
                    "KL", KL.item()
                )

            optimizer.step()

            # collect epoch statistics
            train_loss.append(loss.item())
            train_re.append(RE.item())
            train_kl.append(KL.item())
            train_beta.append(beta)

        epoch_loss = sum(train_loss) / len(train_loader)
        epoch_re = sum(train_re) / len(train_loader)
        epoch_kl = sum(train_kl) / len(train_loader)

        end_epoch_time = time.time()
        epoch_time_diff = end_epoch_time - start_epoch_time

        print(f"Epoch: {epoch}; loss: {epoch_loss}, RE: {epoch_re}, KL: {epoch_kl}, time elapsed: {epoch_time_diff}")
        # add values per batch to epoch
        train_loss_per_epoch.append(epoch_loss)
        train_re_per_epoch.append(epoch_re)
        train_kl_per_epoch.append(epoch_kl)

        # save parameters
        with open(file_name, "wb") as f:
            torch.save(model, f)

    # store loss-values per epoch for plotting
    filename = '{}_{}_lossvalues_train.json'.format(config['dataset_name'], config['prior'])
    loss_values_per_epoch = {
        'model_name': filename,
        "train_loss": train_loss_per_epoch,
        "train_re": train_re_per_epoch,
        "train_kl": train_kl_per_epoch,
        "number_epochs":config['max_epoch'],
        "prior":config['prior'],
        "pseudo_components":config['pseudo_components'],
        "learning_rate":config['learning_rate'],
        "hidden_components":config['z1_size'],
    }


    with open('plots/{}'.format(filename), 'w+') as fp:
        json.dump(loss_values_per_epoch, fp)


def testing(model, train_loader, test_loader, config):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for i, data in enumerate(test_loader):
        # get input, data as the list of [inputs, label]
        inputs, _ = data
        inputs = inputs.to(device)

        mean_dec, logvar_dec, z, mean_enc, logvar_enc = \
            model.forward(inputs)
        loss, RE, KL = model.get_loss(
            inputs, mean_dec, z, mean_enc, logvar_enc, beta=1.0
        )

        test_loss.append(loss.item())
        test_re.append(RE.item())
        test_kl.append(KL.item())

    mean_loss = sum(test_loss) / len(test_loader)
    mean_re = sum(test_re) / len(test_loader)
    mean_kl = sum(test_kl) / len(test_loader)

    # store loss-values for plotting
    filename = '{}_{}_lossvalues_test.json'.format(config['dataset_name'], config['prior'])
    loss_values_per_batch = {
        'model_name': filename,
        "test_loss": test_loss,
        "test_re": test_re,
        "test_kl": test_kl,
        "number_epochs":config['max_epoch'],
        "prior":config['prior'],
        "pseudo_components":config['pseudo_components'],
        "learning_rate":config['learning_rate'],
        "hidden_components":config['z1_size'],
    }

    with open('plots/{}'.format(filename), 'w+') as fp:
        json.dump(loss_values_per_batch, fp)

    print(f"Test results: loss avg: {mean_loss}, RE avg: {mean_re}, KL: {mean_kl}")


def mnist(config):
    torch.manual_seed(14)

    train_loader, eval_loader, test_loader = load_static_mnist(config)

    # If a snapshot exist in /snapshots then use trained weights
    file_name = config["file_name_model"]
    model = VAE(config)
    model.to(device)

    print("Starting training")
    start_time = time.time()

    max_epoch = config["max_epoch"]
    warmup = config["warmup"]
    learning_rate = config["learning_rate"]
    training(
        model,
        train_loader, 
        max_epoch, 
        warmup, 
        file_name,
        config,
        learning_rate
    )
    end_time = time.time()
    time_diff = end_time - start_time
    print("--> Training done, time elapsed: ", time_diff)
    print("--> Testing on test data")
    testing(
        model,
        train_loader, 
        test_loader,
        config
    )
    generate(config["file_name_model"], config["input_size"])


if __name__ == "__main__":
    mnist(config)
