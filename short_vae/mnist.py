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

class Experiment():
    def __init__(self, epochs, batch_size, lr, transforms, training_data, validation_data):
        self.epochs = epochs
        self.batch_size = batch_size
        self.transforms = transforms
        self.train_data = training_data
        self.val_data = validation_data
        self.lr = lr


def mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        root="../input/data", train=True, download=True, transform=transform
    )
    val_data = datasets.MNIST(
        root="../input/data", train=False, download=True, transform=transform
    )
    run(Experiment(20, 64, 0.0001, transform, train_data, val_data))

mnist()
