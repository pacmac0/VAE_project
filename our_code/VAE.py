import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from nn_utils import init_weights, he_init
from distribution_helpers import (
    log_Normal_standard,
    log_Normal_diag,
    log_Logistic_256,
    log_Bernoulli,
)

class NonLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None):
        super(NonLinear, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(int(in_dim), int(out_dim), bias=bias)

    def forward(self, x):
        hh = self.linear(x)
        if self.activation is not None:
            hh = self.activation(hh)
        return hh


class GatedDense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedDense, self).__init__()
        self.h = nn.Linear(int(in_dim), int(out_dim))
        self.g = nn.Linear(int(in_dim), int(out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.h(x) * self.sigmoid(self.g(x))


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.n_hidden = 300

        # encoder q(z|x)
        self.encoder = nn.Sequential(
            GatedDense(np.prod(self.args["input_size"]), self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )
        self.z_mean = nn.Linear(self.n_hidden, self.args["z1_size"])
        # TODO: why use these min,max values?
        self.z_logvar = NonLinear(
            self.n_hidden,
            self.args["z1_size"],
            activation=nn.Hardtanh(min_val=-6.0, max_val=2.0),
        )

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            GatedDense(self.args["z1_size"], self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )
        self.p_mean = NonLinear(
            self.n_hidden, np.prod(self.args["input_size"]), activation=nn.Sigmoid()
        )
        # TODO: why use these min,max values?
        self.p_logvar = NonLinear(
            self.n_hidden,
            np.prod(self.args["input_size"]),
            activation=nn.Hardtanh(min_val=-4.5, max_val=0),
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    # re-parameterization
    def sample_z(self, mean, logvar):
        e = Variable(torch.randn(self.args["z1_size"]))  # ~ N(0,1)
        stddev = torch.exp(logvar / 2)
        return mean + stddev * e

    # Forward through the whole VAE
    def forward(self, x):
        xh = self.encoder(x)
        mean_enc = self.z_mean(xh)
        logvar_enc = self.z_logvar(xh)
        z = self.sample_z(mean_enc, logvar_enc)
        zh = self.decoder(z)
        return self.p_mean(zh), self.p_logvar(zh), z, mean_enc, logvar_enc

    # Loss function: -rec.err + beta*KL-div
    def get_loss(self, x, mean_dec, z, mean_enc, logvar_enc, beta=1):
        re = log_Bernoulli(x, mean_dec, dim=1)
        log_prior = log_Normal_standard(z, dim=1)  # TODO: vampprior
        log_dec_posterior = log_Normal_diag(z, mean_enc, logvar_enc, dim=1)
        kl = -(log_prior - log_dec_posterior)
        l = -re + beta * kl
        return torch.mean(l), torch.mean(re), torch.mean(kl)

    def generate_x(self, N=25):
        z_sample_rand = Variable(torch.FloatTensor(N, self.args["z1_size"]).normal_())
        z = self.decoder(z_sample_rand)
        x_mean = self.p_mean(z)
        return x_mean

def training(model, train_loader, max_epoch, warmup, file_name, 
        learning_rate=0.0005):

    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, max_epoch):
        # Warm up
        beta = 1.0 * epoch / warmup
        if beta > 1.0:
            beta = 1.0
        print(f"--> beta: {beta}")

        train_loss = []
        train_re = []
        train_kl = []

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # print("\nTraining batch #", i)

            # get input, data as the list of [inputs, label]
            inputs, _ = data
            inputs = inputs.to(dev)

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

        epoch_loss = sum(train_loss) / len(train_loader)
        epoch_re = sum(train_re) / len(train_loader)
        epoch_kl = sum(train_kl) / len(train_loader)

        print(f"Epoch: {epoch}; loss: {epoch_loss}, RE: {epoch_re}, KL: {epoch_kl}")

        # save parameters
        with open(file_name, "wb") as f:
            torch.save(model, f)

