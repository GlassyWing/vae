import torch.nn as nn
import torch


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            Swish())

    def forward(self, x):
        return x + self.seq(x)
