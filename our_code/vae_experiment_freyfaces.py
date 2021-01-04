#!/usr/bin/env python3
from scipy.io import loadmat
import torch

# DOWNLOAD FROM HERE: http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat
def freyfaces():
    path = "datasets/freyfaces/frey_rawface.mat"
    batch_size = 64

    ff = loadmat(path)
    ff = ff["ff"].T.reshape((-1, 1, 28, 20)).astype('float32')/255.
    ff = ff[:int(len(ff)/batch_size)*batch_size]
    ff_torch = torch.from_numpy(ff)

    train_size = 1565
    test_size = 200
    train = ff_torch[:train_size]
    test = ff_torch[train_size:train_size+test_size]
    val = ff_torch[train_size+test_size:]

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)

    return train_loader, test_loader, val_loader
