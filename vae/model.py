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


def fit(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)  # = model.forward(data)
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
