import datetime
import os.path as osp
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.utils.data as data_utils
from VAE import VAE, training

config = {
    "seed": 14,
    "dataset_name": "static_mnist",
    "model_name": "vae",
    "prior": "standard",
    "number_components": 500,
    "warmup": 100,
    "z1_size": 40,
    "z2_size": 40,
    "batch_size": 100,
    "test_batch_size": 100,
    "input_size": [1, 28, 28],
    "input_type": "binary",
    "dynamic_binarization": False,
    "use_training_data_init": 1,
    "pseudoinputs_std": 0.01,
    "pseudoinputs_mean": 0.05,
    "learning_rate": 0.0005,
    "max_epoch": 2000,
    "file_name_model": "./snapshots/model.model",
}

# use GPU
if torch.cuda.is_available():
    dev = 'cuda'
    print("--> Using GPU Cuda")
else:
    dev = 'cpu'
    torch.set_num_threads(8) # threading on cpu only
    print("--> Using CPU")

device = torch.device(dev) 

def load_static_mnist(args):
    args["input_size"] = [1, 28, 28]
    args["input_type"] = "binary"
    args["dynamic_binarization"] = False
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
        batch_size=args["batch_size"], shuffle=True)

    eval_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.from_numpy(evaluation_data), torch.from_numpy(evaluation_labels)), 
        batch_size=args["test_batch_size"], shuffle=False)

    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels)), 
        batch_size=args["test_batch_size"], shuffle=True)

    # setting pseudo-inputs inits, for use of vamp-prior
    """
    if args["use_training_data_init"] == 1:
        args["pseudoinputs_std"] = 0.01
        init = train_data[0 : args["number_components"]].T
        args["pseudoinputs_mean"] = torch.from_numpy(
            init
            + args["pseudoinputs_std"]
            * np.random.randn(np.prod(args["input_size"]), args["number_components"])
        ).float()
    else:
        args["pseudoinputs_mean"] = 0.05
        args["pseudoinputs_std"] = 0.01
    """
    return train_loader, eval_loader, test_loader, args

# usage from main: plot_tensor(inputs[0])
def plot_tensor(tensor):
    nparr = tensor.numpy()
    img = np.reshape(nparr, (28, 28))
    plt.figure()
    plt.imshow(img)
    plt.show()


def main(args):
    torch.manual_seed(14)
    model_name = (
        args["dataset_name"]
        + "_"
        + args["model_name"]
        + "_"
        + args["prior"]
        + "(K_"
        + str(args["number_components"])
        + ")"
        + "_wu("
        + str(args["warmup"])
        + ")"
        + "_z1_"
        + str(args["z1_size"])
        + "_z2_"
        + str(args["z2_size"])
    )
    print(args)

    
    # TODO: refactor load_static_mnist
    train_loader, eval_loader, test_loader, args = load_static_mnist(args)

    # If a snapshot exist in /snapshots then use trained weights
    file_name = args["file_name_model"]
    if osp.exists(file_name):
        with open(file_name, "rb") as f:
            model = torch.load(f)
        print("--> Loaded from pretrained model")
    else:  # Otherwise create and intialize a new model
        model = VAE(args)
        print("--> Initialized new model")
    model.to(device)

    max_epoch = args["max_epoch"]
    warmup = args["warmup"]
    learning_rate = args["learning_rate"]
    training(
        model,
        train_loader, 
        max_epoch, 
        warmup, 
        file_name, 
        learning_rate
    )
    
if __name__ == "__main__":
    main(config)