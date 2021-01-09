import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import time
from collections import OrderedDict
import math
import jsonpickle
import random
from distribution_helpers import (
    log_Normal_standard,
    log_Normal_diag,
    log_Bernoulli,
    log_Logistic_256,
)
from eval_generate import generate


class Logger:
    def __init__(self, config):
        self.config = config

        # Each entry corresponds to the average in 1 epoch.
        self.testloss = []
        self.testre = []
        self.testkl = []

        self.trainloss = []
        self.trainre = []
        self.trainkl = []


    def add_test_epoch(self, loss, re, kl):
        self.testloss.append(loss)
        self.testre.append(re)
        self.testkl.append(kl)

    def add_train_epoch(self, loss, re, kl):
        self.trainloss.append(loss)
        self.trainre.append(re)
        self.trainkl.append(kl)


    def dump(self):
        filename = (
            f'experiments/{self.config["dataset_name"]}/{self.config["prior"]}/log'
        )
        json = jsonpickle.encode(self)
        with open(filename, "w+") as f:
            f.write(json)


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
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        self.n_hidden = 300

        # encoder q(z|x)
        self.encoder = nn.Sequential(
            GatedDense(np.prod(self.config["input_size"]), self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )
        self.z_mean = nn.Linear(self.n_hidden, self.config["z1_size"])

        # Choice of interval made to avoid variance to converge to zero
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

        # OBS: we're not using this for binary data
        # Choice of interval made to avoid variance to converge to zero
        self.p_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.config["input_size"]), bias=True),
            nn.Hardtanh(min_val=-4.5, max_val=0),
        )

        # initialise weights for linear layers, not activations
        # https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_train_deep_feedforward_neural_networks [12], eq. (16).
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        # init a layer that will learn pseudos
        # activation with hardtanh 0 to 1 to squeeze data to interval
        if self.config["prior"] == "vamp":
            self.pseudos = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear",
                            nn.Linear(
                                self.config["pseudo_components"],
                                np.prod(self.config["input_size"]),
                                bias=False,
                            ),
                        ),
                        ("activation", nn.Hardtanh(min_val=0, max_val=1)),
                    ]
                )
            )
            # an identity matrix that represents a one-hot encoding for
            # the pseudo-data, where backprop stops.
            self.gradient_start = Variable(
                torch.eye(
                    self.config["pseudo_components"], self.config["pseudo_components"]
                ),
                requires_grad=False,
            ).to(config["device"])
            # init pseudo layer
            if config["pseudo_from_data"]:
                self.pseudos.linear.weight.data = self.config["pseudo_mean"]
            else:  # just set them from arguments
                self.pseudos.linear.weight.data.normal_(
                    self.config["pseudo_mean"], self.config["pseudo_std"]
                )

        if self.config["prior"] == "mog":
            # TODO: tune interval for activation functions
            self.mog_means = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear",
                            nn.Linear(
                                self.config["pseudo_components"],
                                np.prod(self.config["z1_size"]),
                                bias=False,
                            ),
                        ),
                        ("activation", nn.Hardtanh(min_val=0, max_val=1)),
                    ]
                )
            )
            self.mog_logvar = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear",
                            nn.Linear(
                                self.config["pseudo_components"],
                                np.prod(self.config["z1_size"]),
                                bias=False,
                            ),
                        ),
                        ("activation", nn.Hardtanh(min_val=-2, max_val=2)),
                    ]
                )
            )
            # an identity matrix that represents a one-hot encoding for
            # the each parameter pair, where backprop stops.
            self.gradient_start = Variable(
                torch.eye(
                    self.config["pseudo_components"], self.config["pseudo_components"]
                ),
                requires_grad=False,
            ).to(config["device"])

            # just set them from arguments
            # self.mog_means.linear.weight.data.normal_(self.config['pseudo_mean'], self.config['pseudo_std'])
            # self.mog_logvar.linear.weight.data.normal_(self.config['pseudo_mean'], self.config['pseudo_std'])
            # TODO: should we use xavier init here?
            torch.nn.init.xavier_uniform_(self.mog_means.linear.weight)
            torch.nn.init.xavier_uniform_(self.mog_logvar.linear.weight)

    # re-parameterization
    # enough to sample one point if batches large enough,
    # 2.4 https://arxiv.org/pdf/1312.6114.pdf
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
        if self.config["prior"] == "vamp":
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
            K = self.config["pseudo_components"]
            a = log_Normal_diag(z, means, logvars, dim=2) - math.log(K)
            b, _ = torch.max(a, 1)
            log_prior = b + torch.log(torch.sum(torch.exp(a - b.unsqueeze(1)), 1))
            return log_prior
        elif self.config["prior"] == "mog":
            mean = self.mog_means(self.gradient_start)
            logvar = self.mog_logvar(self.gradient_start)

            z = z.unsqueeze(1)
            mean = mean.unsqueeze(0)
            logvar = mean.unsqueeze(0)

            if self.config['input_type'] == 'binary':
                logs = log_Bernoulli(z, mean, dim=1)
            else:
                logs = log_Normal_diag(z, mean, logvar, dim=1)
            s = torch.sum(torch.exp(logs))
            K = self.config["pseudo_components"]
            return torch.log(s / K)
        else:  # std gaussian
            return log_Normal_standard(z, dim=1)

    # Loss function: -rec.err + beta*KL-div
    def get_loss(self, x, mean_enc, logvar_enc, z, mean_dec, logvar_dec, beta=1):
        # Different types of data have different likelihoods
        # Appendix C, https://arxiv.org/pdf/1312.6114.pdf
        if self.config["input_type"] == "binary":
            re = log_Bernoulli(x, mean_dec, dim=1)
        elif self.config["input_type"] == "cont":
            re = -log_Logistic_256(x, mean_dec, logvar_dec, dim=1)
        else:
            raise Exception("Input type unknown")

        log_prior = self.get_log_prior(z)
        log_dec_posterior = log_Normal_diag(z, mean_enc, logvar_enc, dim=1)
        kl = -(log_prior - log_dec_posterior)
        l = -re + beta * kl
        return torch.mean(l), torch.mean(re), torch.mean(kl)

    def generate_samples(self, N=25):
        if self.config["prior"] == "vamp":
            # sample N learned pseudo-data
            pseudos = self.pseudos(self.gradient_start)[0:N]
            # put through encoder
            ps_h = self.encoder(pseudos)
            ps_mean_enc = self.z_mean(ps_h)
            ps_logvar_enc = self.z_logvar(ps_h)
            # re-param
            z_samples = self.sample_z(ps_mean_enc, ps_logvar_enc)
        elif self.config["prior"] == "mog":
            mean = self.mog_means(self.gradient_start)[0:N]
            if self.config['input_type'] == 'binary':
                z_samples = torch.bernoulli(mean)
            else: 
                logvar = self.mog_logvar(self.gradient_start)[0:N]
                z_samples = self.sample_z(mean, logvar)
        else:  # standard prior
            # sample N latent points from std gaussian prior
            z_samples = Variable(
                torch.FloatTensor(N, self.config["z1_size"]).normal_()
            ).to(self.config["device"])

        # decode and use means sample data
        z = self.decoder(z_samples)
        x_mean = self.p_mean(z)
        return x_mean

    def get_pseudos(self, N=25):
        # pick random pseudo-inputs
        start = random.randint(0, self.config["pseudo_components"] - 1 - N)
        return self.pseudos(self.gradient_start)[start : start + N]
        # return self.pseudos(self.gradient_start)[0:N]


def train(model, train_loader, config, test_loader):
    logger = Logger(config)

    torch.autograd.set_detect_anomaly(True)  # trace err if nan values
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / config["warmup"]
        if beta > 1.0:
            beta = 1.0

        train_loss = 0
        train_re = 0
        train_kl = 0

        start_epoch_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # get input, data as the list of [data, label]

            if config["dataset_name"] == "mnist":
                data, _ = data
            data = data.to(config["device"])

            # forward
            mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(data)

            # calculate loss
            loss, RE, KL = model.get_loss(
                data, mean_enc, logvar_enc, z, mean_dec, logvar_dec, beta=beta
            )
            # backpropagate
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_re += RE.item()
            train_kl += KL.item()

        epoch_loss = train_loss / config["batch_size"]
        epoch_re = train_re / config["batch_size"]
        epoch_kl = train_kl / config["batch_size"]
        logger.add_train_epoch(epoch_loss, epoch_re, epoch_kl)

        end_epoch_time = time.time()
        epoch_time_diff = end_epoch_time - start_epoch_time

        print(
            f"{config['dataset_name']}, {config['prior']}, epoch: {epoch}: loss: {epoch_loss:.3f}, RE: {epoch_re:.3f}, KL: {epoch_kl:.3f}, time elapsed: {epoch_time_diff:.3f}"
        )

        test(model, test_loader, config, logger)
        logger.dump()

        # save parameters
        if epoch % 20 == 0:
            generate(model, config, epoch)


def test(model, test_loader, config, logger):
    test_loss = []
    test_re = []
    test_kl = []

    # evaulation mode
    model.eval()

    for data in test_loader:
        # get input, data as the list of [data, label]
        if config["dataset_name"] == "mnist":
            data, _ = data
        else:
            data = data
        data = data.to(config["device"])

        mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(data)
        loss, RE, KL = model.get_loss(
            data, mean_enc, logvar_enc, z, mean_dec, logvar_dec, beta=1.0
        )

        test_loss.append(loss.item())
        test_re.append(RE.item())
        test_kl.append(KL.item())

    mean_loss = sum(test_loss) / len(test_loader)
    mean_re = sum(test_re) / len(test_loader)
    mean_kl = sum(test_kl) / len(test_loader)

    logger.add_test_epoch(mean_loss, mean_re, mean_kl)
    print(
        f"Test results: loss avg: {mean_loss:.3f}, RE avg: {mean_re:.3f}, KL: {mean_kl:.3f}"
    )


def add_pseudo_prior(config, train_data):
    # get pseudo init params from random data
    # and add some randomness so it is not the exactly the same
    if config["pseudo_from_data"] and config["prior"] == "vamp":
        config["pseudo_std"] = 0.01
        np.random.shuffle(train_data)
        # print("DIM: {}".format(train_data.shape))
        dat = train_data[
            0 : int(config["pseudo_components"])
        ].T  # make columns components(data-points)
        # print("DIM: {}".format(dat.shape))
        # add some randomness to the pseudo inputs to avoid overfitting
        rand_std_norm = np.random.randn(
            np.prod(config["input_size"]), config["pseudo_components"]
        )
        config["pseudo_mean"] = torch.from_numpy(
            dat + config["pseudo_std"] * rand_std_norm
        ).float()
    else:
        # TODO: fine tune more? Maybe not same for frey and MNIST
        config["pseudo_std"] = 0.01
        config["pseudo_mean"] = 0.05
