import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from nn_utils import init_weights, he_init
from distribution_helpers import log_Normal_standard, log_Normal_diag, log_Logistic_256, log_Bernoulli

# Taken from VAE paper
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

# Taken from VAE paper
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.n_hidden = 300

        # TODO check different layers of paper
        # encoder q(z|x)
        self.encoder = nn.Sequential(
            GatedDense(np.prod(self.args['input_size']), self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden)
        )
        self.z_mean = nn.Linear(self.n_hidden, self.args['z1_size'])
        self.z_logvar = NonLinear(self.n_hidden, self.args['z1_size'], activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            GatedDense(self.args['z1_size'], self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden)
        )
        self.p_mean = NonLinear(self.n_hidden, np.prod(self.args['input_size']), activation=nn.Sigmoid())
        self.p_logvar = NonLinear(self.n_hidden, np.prod(self.args['input_size']), activation=nn.Hardtanh(min_val=-4.5,max_val=0))


        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)


    # re-parameterization
    def sample_z(self, mean, logvar):
        e = Variable(torch.randn(self.args['z1_size'])) # ~ N(0,1)
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
        # self.mean_dec, self.logvar_dec, self.z, self.mean_enc, self.logvar_enc = mean_dec, logvar_dec, z, mean_enc, logvar_enc
        re = log_Bernoulli(x, mean_dec, dim=1)
        # self.re = -log_Logistic_256(x, self.mean_dec, self.logvar_dec, dim=1) # TODO: make usable for other dimensions
        log_prior = log_Normal_standard(z, dim=1) # TODO: exchange with vampprior
        log_dec_posterior = log_Normal_diag(z, mean_enc, logvar_enc, dim=1)
        kl = -(log_prior - log_dec_posterior)
        l = -re + beta*kl
        return torch.mean(l), torch.mean(re), torch.mean(kl) # TODO: do we need to return everything?

    def generate_x(self, N=25):
        z_sample_rand = Variable(torch.FloatTensor(N, self.args['z1_size']).normal_())

        z = self.decoder(z_sample_rand)
        x_mean = self.p_mean(z)

        return x_mean

    # Other methods for inspecting the model
    def get_latent(self, x):
        x_var = Variable(x)
        xh = self.encoder(x_var)
        z_mean = self.z_mean(xh)
        z_logvar = self.z_logvar(xh)

        return z_mean, z_logvar
