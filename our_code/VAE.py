import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from distribution_helpers import log_Normal_standard, log_Normal_diag, log_Logistic_256, log_Bernoulli

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.n_hidden = 300


        # TODO check different layers of paper
        # encoder q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.args['input_size']), self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden)
        )
        self.z_mean = nn.Linear(self.n_hidden, self.args['z1_size'])
        self.z_logvar = nn.Linear(self.n_hidden, self.args['z1_size'])

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.args['z1_size'], self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden)
        )
        self.p_mean = nn.Linear(self.n_hidden, np.prod(self.args['input_size']))
        self.p_logvar = nn.Linear(self.n_hidden, np.prod(self.args['input_size']))

        # initialise the weights # TODO check different init types
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        # TODO check deprecated function xavier_uniform
        self.encoder.apply(init_weights)
        torch.nn.init.xavier_uniform(self.z_mean.weight)
        torch.nn.init.xavier_uniform(self.z_logvar.weight)
        self.decoder.apply(init_weights)
        torch.nn.init.xavier_uniform(self.p_mean.weight)
        torch.nn.init.xavier_uniform(self.p_logvar.weight)

    # re-parameterization
    def sample_z(self, mean, logvar):
        e = Variable(torch.randn(self.args['z1_size'])) # ~ N(0,1)
        stddev = torch.exp(logvar / 2)
        return mean + stddev * e

    # Forward through the whole VAE
    def forward(self, x):
        self.xh = self.encoder(x)
        self.mean_enc = self.z_mean(self.xh)
        self.logvar_enc = self.z_logvar(self.xh)
        self.z = self.sample_z(self.mean_enc,self.logvar_enc)
        self.zh = self.decoder(self.z)
        self.mean_dec = self.p_mean(self.zh)
        self.logvar_dec = self.p_logvar(self.zh)
        return self.mean_dec, self.logvar_dec, self.z, self.mean_enc, self.logvar_enc

    # Loss function: -rec.err + beta*KL-div
    def get_loss(self, x, mean_dec, logvar_dec, z, mean_enc, logvar_enc, beta=1):
        # self.mean_dec, self.logvar_dec, self.z, self.mean_enc, self.logvar_enc = mean_dec, logvar_dec, z, mean_enc, logvar_enc
        self.re = -log_Bernoulli(x, self.mean_dec, dim=1)
        print('re', self.re)
        # self.re = -log_Logistic_256(x, self.mean_dec, self.logvar_dec, dim=1) # TODO: make usable for other dimensions
        
        self.log_prior = log_Normal_standard(z, average=True, dim=1) # TODO: exchange with vampprior
        print('log_prior', self.log_prior)
        self.log_dec_posterior = log_Normal_diag(z, self.mean_enc, self.logvar_enc, average=True, dim=1)
        print('log_dec_posterior', self.log_dec_posterior)
        self.kl = -(self.log_prior - self.log_dec_posterior)
        print('kl', self.kl)
        self.l = -self.re + beta*self.kl
        self.loss = torch.mean(self.l)
        self.recon_err = torch.mean(self.re)
        self.kullback = torch.mean(self.kl)
        return self.loss, self.recon_err, self.kullback # TODO: do we need to return everything?
