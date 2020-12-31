#!/usr/bin/env python3
import torch
import torchvision
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
from model import run
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.set_num_threads(8)

epochs = 20
batch_size = 64
lr = 0.0001


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(
    root="../input/data", train=True, download=True, transform=transform
)

val_data = datasets.MNIST(
    root="../input/data", train=False, download=True, transform=transform
)



run(epochs, batch_size, lr, train_data, val_data)
