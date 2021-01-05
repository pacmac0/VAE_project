import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import time
import os
import torch.utils.data as data_utils
from collections import OrderedDict 
import math
import json
from distribution_helpers import (
    log_Normal_standard,
    log_Normal_diag,
    log_Logistic_256,
    log_Bernoulli,
)


# https://arxiv.org/abs/1612.08083 [8], eq. (1), ch.2
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
        self.config = args
        self.n_hidden = 300

        # encoder q(z|x)
        self.encoder = nn.Sequential(
            GatedDense(np.prod(self.config["input_size"]), self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )
        self.z_mean = nn.Linear(self.n_hidden, self.config["z1_size"])

        # TODO: play around with min max vals for hardtanh
        self.z_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, self.config["z1_size"], bias=True),
            nn.Hardtanh(min_val=-6, max_val=2),
        )

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            GatedDense(self.config["z1_size"], self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )

        # the mean needs to be a probability since mnist binary data
        # gives bernoulli, so use sigmoid activation instead
        self.p_mean = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.config["input_size"]), bias=True),
            nn.Sigmoid(),
        )

        # OBS: we're not using this for discrete data
        self.p_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.config["input_size"]), bias=True),
            nn.Hardtanh(min_val=-4.5, max_val=0),
        )

        # init a layer that will learn pseudos
        if self.config["prior"] == "vamp":
            self.pseudos = nn.Sequential(
                nn.Linear(self.config["pseudo_components"],
                    np.prod(self.config["input_size"]), bias=False),
                nn.Hardtanh(min_val=0, max_val=1)
                )

        if self.config['prior'] == 'mog':
            self.mog_means = nn.Sequential(
                nn.Linear(self.config["pseudo_components"],
                    np.prod(self.config["input_size"]), bias=False),
                nn.Hardtanh(min_val=0, max_val=1)
                )
            self.mog_logvar =  nn.Sequential(
                nn.Linear(self.config["pseudo_components"],
                    np.prod(self.config["input_size"]), bias=False),
                nn.Hardtanh(min_val=-6, max_val=2)
                )
        # initialise weights for linear layers, not activations
        # https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_training_deep_feedforward_neural_networks [12], eq. (16).
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
             
        # init a layer that will learn pseudos
        # activation with hardtanh 0 to 1 to squeeze data to interval
        if self.config["prior"] == "vamp":
            self.pseudos = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.config["pseudo_components"],np.prod(self.config["input_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            # an identity matrix that represents a one-hot encoding for
            # the pseudo-data, where backprop stops.
            self.gradient_start = Variable(torch.eye(self.config["pseudo_components"], self.config["pseudo_components"]), requires_grad=False).to(config["device"])
            # init pseudo layer
            if args['pseudo_from_data']:
                self.pseudos.linear.weight.data = self.config['pseudo_mean']
            else: # just set them from arguments
                self.pseudos.linear.weight.data.normal_(self.config['pseudo_mean'], self.config['pseudo_std'])
                #torch.nn.init.xavier_uniform_(self.pseudos.linear.weight)

        if self.config["prior"] == "mog":
            self.mog_means = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.config["pseudo_components"],np.prod(self.config["z1_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            self.mog_logvar = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.config["pseudo_components"],np.prod(self.config["z1_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            # an identity matrix that represents a one-hot encoding for
            # the pseudo-data, where backprop stops.
            self.gradient_start = Variable(torch.eye(self.config["pseudo_components"], self.config["pseudo_components"]), requires_grad=False).to(config["device"])
            
            # just set them from arguments
            # self.mog_means.linear.weight.data.normal_(self.config['pseudo_mean'], self.config['pseudo_std'])
            # self.mog_logvar.linear.weight.data.normal_(self.config['pseudo_mean'], self.config['pseudo_std'])
            torch.nn.init.xavier_uniform_(self.mog_means.linear.weight)
            torch.nn.init.xavier_uniform_(self.mog_logvar.linear.weight)
    
    # re-parameterization
    def sample_z(self, mean, logvar):
        e = Variable(torch.randn(self.config["z1_size"])).to(self.config["device"])
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

    def get_log_prior(self, z):
        if self.config['prior'] == 'vamp':
            pseudos = self.pseudos(self.gradient_start)
            xh = self.encoder(pseudos)
            # encoded pseudos
            pseudo_means = self.z_mean(xh)
            pseudo_logvars = self.z_logvar(xh)

            # squeeze stuff to correct dims
            # to match with the batch size for z
            z = z.unsqueeze(1)
            means = pseudo_means.unsqueeze(0)
            logvars = pseudo_logvars.unsqueeze(0)

            # sum togther variational posteriors, eq. (9)
            # using log-sum-exp trick to avoid underflow
            # http://wittawat.com/posts/log-sum_exp_underflow.html
            K = self.config['pseudo_components']
            a = log_Normal_diag(z, means, logvars, dim=2) - math.log(K)
            b, _ = torch.max(a, 1) 

            # calculte log-sum-exp
            log_prior = b + torch.log(
                torch.sum(torch.exp(a - b.unsqueeze(1)), 1)) 
            return log_prior
        elif self.config['prior'] == 'mog':
            mean = self.mog_means(self.gradient_start)
            logvar = self.mog_logvar(self.gradient_start)

            z = z.unsqueeze(1)
            mean = mean.unsqueeze(0)
            logvar = mean.unsqueeze(0)

            logs  = log_Normal_diag(z, mean, logvar, dim=1)
            s = torch.sum(torch.exp(logs))
            K = self.config['pseudo_components']
            return torch.log(s / K)
        else: # std gaussian
            return log_Normal_standard(z, dim=1)

    # Loss function: -rec.err + beta*KL-div
    def get_loss(self, x, mean_dec, z, mean_enc, logvar_enc, beta=1):
        re = log_Bernoulli(x, mean_dec, dim=1)
        log_prior = self.get_log_prior(z)
        log_dec_posterior = log_Normal_diag(z, mean_enc, logvar_enc, dim=1)
        kl = -(log_prior - log_dec_posterior)
        l = -re + beta * kl
        return torch.mean(l), torch.mean(re), torch.mean(kl)

    def generate_samples(self, N=25):
        print('prior', self.config['prior'])
        if self.config['prior'] == 'vamp':
            # sample N learned pseudo-data
            pseudos = self.pseudos(self.gradient_start)[0:N]
            # put through encoder
            ps_h = self.encoder(pseudos)
            ps_mean_enc = self.z_mean(ps_h)
            ps_logvar_enc = self.z_logvar(ps_h)
            # re-param
            z_samples = self.sample_z(ps_mean_enc, ps_logvar_enc)
        elif self.config['prior'] == 'mog':
            mean = self.mog_means(self.gradient_start)[0:N]
            logvar = self.mog_logvar(self.gradient_start)[0:N]            
            z_samples = self.sample_z(mean, logvar)
        else: # standard prior
            # sample N latent points from std gaussian prior
            z_samples = Variable(torch.FloatTensor(N, self.config["z1_size"]).normal_()).to(self.config["device"])

        # decode and use means sample data
        z = self.decoder(z_samples)
        x_mean = self.p_mean(z)
        return x_mean


def training(model, train_loader, config,
        learning_rate=0.0005):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    train_re_per_epoch = []
    train_kl_per_epoch = []
    for epoch in range(1, config["epochs"]+1):
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / config["warmup"]
        if beta > 1.0:
            beta = 1.0
        print(f"beta: {beta}")

        train_loss = []
        train_re = []
        train_kl = []
        train_beta = []

        start_epoch_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # get input, data as the list of [data, label]

            if config["dataset_name"] == "static_mnist":
                data, _ = data
            data = data.to(config["device"])

            # forward
            mean_dec, logvar_dec, z, mean_enc, logvar_enc = \
                model.forward(data)

            # calculate loss
            loss, RE, KL = model.get_loss(
                data, mean_dec, z, mean_enc, logvar_enc, beta=beta
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
        with open(config["file_name_model"], "wb") as f:
            torch.save(model, f)

    # store loss-values per epoch for plotting
    filename = '{}_{}_lossvalues_train.json'.format(config['dataset_name'], config['prior'])
    loss_values_per_epoch = {
        'model_name': filename,
        "train_loss": train_loss_per_epoch,
        "train_re": train_re_per_epoch,
        "train_kl": train_kl_per_epoch,
        "number_epochs":config["epochs"],
        "prior":config['prior'],
        "pseudo_components":config['pseudo_components'],
        "learning_rate":config['learning_rate'],
        "hidden_components":config['z1_size'],
    }


    with open('plots/{}'.format(filename), 'w+') as fp:
        json.dump(loss_values_per_epoch, fp)


def testing(model, test_loader, config):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for i, data in enumerate(test_loader):
        # get input, data as the list of [data, label]
        if config["dataset_name"] == "static_mnist":
            data, _ = data
        else:
            data = data
        data = data.to(config["device"])

        mean_dec, logvar_dec, z, mean_enc, logvar_enc = \
            model.forward(data)
        loss, RE, KL = model.get_loss(
            data, mean_dec, z, mean_enc, logvar_enc, beta=1.0
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
        "number_epochs":config['epochs'],
        "prior":config['prior'],
        "pseudo_components":config['pseudo_components'],
        "learning_rate":config['learning_rate'],
        "hidden_components":config['z1_size'],
    }

    with open('plots/{}'.format(filename), 'w+') as fp:
        json.dump(loss_values_per_batch, fp)

    print(f"Test results: loss avg: {mean_loss}, RE avg: {mean_re}, KL: {mean_kl}")
