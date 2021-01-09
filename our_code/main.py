#!/usr/bin/env python3

from MNIST import mnist
from freyfaces import freyfaces
import os
import torch
import configs
import plot

for prior in ["mog", "vamp", "standard"]:
    for subfolder in ["images", "models"]:
        p = f'experiments/mnist/{prior}/{subfolder}'
        if not os.path.exists(p):
            os.makedirs(p)

    for prior in ["mog", "vamp", "standard"]:
        for subfolder in ["images", "models"]:
            if prior == "vamp":
                for pseudo in ["pseudo_from_data", "not_pseudo_from_data"]:
                    p = f'experiments/freyfaces/{prior}/{pseudo}/{subfolder}'
                    if not os.path.exists(p):
                        os.makedirs(p)
            else:
                p = f'experiments/freyfaces/{prior}/{subfolder}'
                if not os.path.exists(p):
                    os.makedirs(p)

# download http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat to "datasets/freyfaces/frey_rawface.mat"

freyfaces(configs.frey_vamp) # pseudo_from_data = True
plot.plot("experiments/freyfaces/vamp/pseudo_from_data/log","Freyfaces vampprior, with pseudos from data", "Freyfaces_vamp_with_pseudos_from_data", ["train_loss", "test_loss", "test_re", "test_kl", "train_re", "train_kl"])


configs.frey_vamp["pseudo_from_data"] = False
freyfaces(configs.frey_vamp)
plot.plot("experiments/freyfaces/vamp/not_pseudo_from_data/log","Freyfaces vampprior, without pseudos from data", "Freyfaces_vamp_without_pseudos_from_data", ["train_loss", "test_loss", "test_re", "test_kl", "train_re", "train_kl"])

freyfaces(configs.frey_standard)
plot.plot("experiments/freyfaces/standard/log","Freyfaces standard", "Freyfaces_standard", ["train_loss", "test_loss", "test_re", "test_kl", "train_re", "train_kl"])

freyfaces(configs.frey_mog)
plot.plot("experiments/freyfaces/mog/log","Freyfaces mog", "Freyfaces_mog", ["train_loss", "test_loss", "test_re", "test_kl", "train_re", "train_kl"])
