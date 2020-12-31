# https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = 16

class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=784)


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample


    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var


def fit(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    return train_loss


def validate(model, dataloader, batch_size, criterion, val_data, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data) / dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat(
                    (
                        data.view(batch_size, 1, 28, 28)[:8],
                        reconstruction.view(batch_size, 1, 28, 28)[:8],
                    )
                )
                save_image(both.cpu(), f"outputs/output{epoch}.png", nrow=num_rows)
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def final_loss(bce_loss, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + KLD


def run(e):
    model = LinearVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=e.lr)
    criterion = nn.BCELoss(reduction="sum")

    train_loader = DataLoader(e.train_data, batch_size=e.batch_size, shuffle=True)
    val_loader = DataLoader(e.val_data, batch_size=e.batch_size, shuffle=False)
    train_loss = []
    val_loss = []

    for epoch in range(e.epochs):
        print(f"Epoch {epoch+1} of {e.epochs}")
        train_loss.append(fit(model, train_loader, optimizer, criterion))
        val_loss.append(validate(model, val_loader, e.batch_size, criterion, e.val_data, epoch))
        print(f"Train Loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")

        with open("model.model", "wb") as f:
            torch.save(model, f)
