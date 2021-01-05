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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
        self.args = args
        self.n_hidden = 300

        # encoder q(z|x)
        self.encoder = nn.Sequential(
            GatedDense(np.prod(self.args["input_size"]), self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )
        self.z_mean = nn.Linear(self.n_hidden, self.args["z1_size"])

        # TODO: play around with min max vals for hardtanh
        self.z_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, self.args["z1_size"], bias=True),
            nn.Hardtanh(min_val=-6, max_val=2),
        )

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            GatedDense(self.args["z1_size"], self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )

        # the mean needs to be a probability since mnist binary data
        # gives bernoulli, so use sigmoid activation instead
        self.p_mean = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.args["input_size"]), bias=True),
            nn.Sigmoid(),
        )

        # OBS: we're not using this for discrete data
        self.p_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.args["input_size"]), bias=True),
            nn.Hardtanh(min_val=-4.5, max_val=0),
        )

        # init a layer that will learn pseudos
        if self.args["prior"] == "vamp": 
            self.pseudos = nn.Sequential(
                nn.Linear(self.args["pseudo_components"], 
                    np.prod(self.args["input_size"]), bias=False),
                nn.Hardtanh(min_val=0, max_val=1)
                )

        if self.args['prior'] == 'mog':
            self.mog_means = nn.Sequential(
                nn.Linear(self.args["pseudo_components"], 
                    np.prod(self.args["input_size"]), bias=False),
                nn.Hardtanh(min_val=0, max_val=1)
                )
            self.mog_logvar =  nn.Sequential(
                nn.Linear(self.args["pseudo_components"], 
                    np.prod(self.args["input_size"]), bias=False),
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
        if self.args["prior"] == "vamp": 
            self.pseudos = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.args["pseudo_components"],np.prod(self.args["input_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            # an identity matrix that represents a one-hot encoding for
            # the pseudo-inputs, where backprop stops.
            self.gradient_start = Variable(torch.eye(self.args["pseudo_components"], self.args["pseudo_components"]), requires_grad=False).to(device) 
            # init pseudo layer
            if args['pseudo_from_data']:
                self.pseudos.linear.weight.data = self.args['pseudo_mean']
            else: # just set them from arguments
                self.pseudos.linear.weight.data.normal_(self.args['pseudo_mean'], self.args['pseudo_std'])
                #torch.nn.init.xavier_uniform_(self.pseudos.linear.weight)

        if self.args["prior"] == "mog": 
            self.mog_means = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.args["pseudo_components"],np.prod(self.args["z1_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            self.mog_logvar = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.args["pseudo_components"],np.prod(self.args["z1_size"]), bias=False)),
                ('activation', nn.Hardtanh(min_val=0, max_val=1))
            ]))
            # an identity matrix that represents a one-hot encoding for
            # the pseudo-inputs, where backprop stops.
            self.gradient_start = Variable(torch.eye(self.args["pseudo_components"], self.args["pseudo_components"]), requires_grad=False).to(device) 
            
            # just set them from arguments
            # self.mog_means.linear.weight.data.normal_(self.args['pseudo_mean'], self.args['pseudo_std'])
            # self.mog_logvar.linear.weight.data.normal_(self.args['pseudo_mean'], self.args['pseudo_std'])
            torch.nn.init.xavier_uniform_(self.mog_means.linear.weight)
            torch.nn.init.xavier_uniform_(self.mog_logvar.linear.weight)
    
    # re-parameterization
    def sample_z(self, mean, logvar):
        e = Variable(torch.randn(self.args["z1_size"])).to(device)
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
        if self.args['prior'] == 'vamp':
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
            K = self.args['pseudo_components']
            a = log_Normal_diag(z, means, logvars, dim=2) - math.log(K)
            b, _ = torch.max(a, 1) 

            # calculte log-sum-exp
            log_prior = b + torch.log(
                torch.sum(torch.exp(a - b.unsqueeze(1)), 1)) 
            return log_prior
        elif self.args['prior'] == 'mog':
            mean = self.mog_means(self.gradient_start)
            logvar = self.mog_logvar(self.gradient_start)

            z = z.unsqueeze(1)
            mean = mean.unsqueeze(0)
            logvar = mean.unsqueeze(0)

            logs  = log_Normal_diag(z, mean, logvar, dim=1)
            s = torch.sum(torch.exp(logs))
            K = self.args['pseudo_components']
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
        print('prior', self.args['prior'])
        if self.args['prior'] == 'vamp':
            # sample N learned pseudo-inputs
            pseudos = self.pseudos(self.gradient_start)[0:N]
            # put through encoder
            ps_h = self.encoder(pseudos)
            ps_mean_enc = self.z_mean(ps_h)
            ps_logvar_enc = self.z_logvar(ps_h)
            # re-param
            z_samples = self.sample_z(ps_mean_enc, ps_logvar_enc)
        elif self.args['prior'] == 'mog':
            mean = self.mog_means(self.gradient_start)[0:N]
            logvar = self.mog_logvar(self.gradient_start)[0:N]            
            z_samples = self.sample_z(mean, logvar)
        else: # standard prior
            # sample N latent points from std gaussian prior
            z_samples = Variable(torch.FloatTensor(N, self.args["z1_size"]).normal_()).to(device)

        # decode and use means sample data
        z = self.decoder(z_samples)
        x_mean = self.p_mean(z)
        return x_mean

