import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channel_out=3):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channel_out, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, channel_in=3):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(channel_in, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
        





