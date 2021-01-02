import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
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
            nn.Hardtanh(min_val=-4.0, max_val=4.0),
        )

        # decoder p(x|z)
        self.decoder = nn.Sequential(
            GatedDense(self.args["z1_size"], self.n_hidden),
            GatedDense(self.n_hidden, self.n_hidden),
        )

        # the mean needs to be a probability
        # so use sigmoid activation instead
        self.p_mean = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.args["input_size"]), bias=True),
            nn.Sigmoid(),
        )

        # OBS: we're not using this for discrete data
        self.p_logvar = nn.Sequential(
            nn.Linear(self.n_hidden, np.prod(self.args["input_size"]), bias=True),
            nn.Hardtanh(min_val=-2.0, max_val=2.0),
        )

        # initialise weights for linear layers, not activations
        # https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_training_deep_feedforward_neural_networks [12], eq. (16).
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        """
        # pseudo layer
        self.z_logvar = nn.Sequential(
            nn.Linear(self.args['pseudo_components'] * np.prod(self.args["input_size"]), bias=False),
            nn.Hardtanh(min_val=-4.0, max_val=4.0) # check min, max values
        )
        """
        # aprox. pseudo input

    def init_pseudo_inputs(self):
        import os
        import torch.utils.data as data_utils
        with open(
        os.path.join("datasets", "MNIST_static", "binarized_mnist_train.amat")
        ) as f:
            lines = f.readlines()
            train_data = np.array([[int(i) for i in l.split()] for l in lines]).astype("float32")
        train_labels = np.zeros((train_data.shape[0], 1))
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels)), 
            batch_size=self.args["pseudo_components"], shuffle=True)

        for i, data in enumerate(train_loader):
            inputs, _ = data
            inputs = inputs.to(device)
            self.pseudo_inputs = inputs
            return


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

    # Loss function: -rec.err + beta*KL-div
    def get_loss(self, x, mean_dec, z, mean_enc, logvar_enc, beta=1):
        re = log_Bernoulli(x, mean_dec, dim=1)
        log_prior = get_prior(z)
        log_dec_posterior = log_Normal_diag(z, mean_enc, logvar_enc, dim=1)
        kl = -(log_prior - log_dec_posterior)
        l = -re + beta * kl
        return torch.mean(l), torch.mean(re), torch.mean(kl)

    def get_prior(self, z):
        if self.args['prior'] == 'standard':
            return log_Normal_standard(z, dim=1)
        elif self.args['prior'] == 'vamp':
            # put all pseudo-inputs through encoder
            xh = self.encoder(self.pseudo_inputs)
            pseudo_means = self.z_mean(xh)
            pseudo_logvars = self.z_logvar(xh)
            # sum togther variational posteriors, eq. (9)
            logs = log_Normal_diag(z, pseudo_means ,pseudo_logvars, dim=1)
            s =  torch.sum(torch.exp(logs) 
            K = self.args['pseudo_components']
            return s / K # eq. (9)

    def generate_x(self, N=25):
        z_sample_rand = Variable(torch.FloatTensor(N, self.args["z1_size"]).normal_()).to(device)
        z = self.decoder(z_sample_rand)
        x_mean = self.p_mean(z)
        return x_mean

def training(model, train_loader, max_epoch, warmup_period, file_name, 
        learning_rate=0.0005):

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, max_epoch):
        # Warm up
        # https://arxiv.org/abs/1511.06349 [5], KL cost annealing, ch.3.
        beta = 1.0 * epoch / warmup_period
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
            inputs = inputs.to(device)

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

