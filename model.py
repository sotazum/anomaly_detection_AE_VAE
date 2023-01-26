import numpy as np
import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, z_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_h = self.decoder(z)
        return x_h

class VAE(nn.Module):
    def __init__(self, z_dim, device):
        super(VAE, self).__init__()
        self.eps = np.spacing(1)
        self.encoder_layer = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )

        self.encoder_mean = nn.Linear(128, z_dim)
        self.encoder_logvar = nn.Linear(128, z_dim)

        self.decoder_layer = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28)
        )
    
    def encoder(self, x):
        x = self.encoder_layer(x)
        return self.encoder_mean(x), self.encoder_logvar(x) 

    def sampling_z(self, mean, logvar, device):
        ep = torch.randn(mean.shape).to(device)
        return mean + ep * torch.exp(0.5 * logvar)

    def decoder(self, z):
        return torch.sigmoid(self.decoder_layer(z))
    
    def forward(self, x, device):
        mean, logvar = self.encoder(x)
        z = self.sampling_z(mean, logvar, device)
        x_h = self.decoder(z)
        return x_h, z, mean, logvar
