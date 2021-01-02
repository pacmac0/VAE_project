#!/usr/bin/env python3
import torch.nn as nn
import torch

features = 16

class Vampprior(nn.Module):
    def __init__(self):
        super(Vampprior, self).__init__()

        self.enc1 = nn.Linear(in_features=784, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=features * 2)

        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=784)


    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)


    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)

        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance

        z = self.sample_z(mu, log_var)

        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var
