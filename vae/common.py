import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU())

    def forward(self, x):
        return x + self.seq(x)
