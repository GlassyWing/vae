import torch
import torch.nn as nn
import numpy as np

from vae.losses import recon, kl
from vae.utils import reparameterize


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channel, out_channel // 2, kernel_size=1)
        self.conv_3 = nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        return self.leaky_relu(self.bn(x))


class Encoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self._seq = nn.Sequential(
            ConvBlock(3, z_dim // 16),
            ConvBlock(z_dim // 16, z_dim // 8),
            ConvBlock(z_dim // 8, z_dim // 4),
            ConvBlock(z_dim // 4, z_dim // 2),
            ConvBlock(z_dim // 2, z_dim),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self._seq(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channel,
                                           out_channel,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        # x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class Decoder(nn.Module):

    def __init__(self, z_dim, map_size):
        super().__init__()
        self.map_size = map_size
        self.z_dim = z_dim
        self._dense = nn.Linear(z_dim, np.prod(map_size) * z_dim)
        self._seq = nn.Sequential(
            UpsampleBlock(z_dim, z_dim // 2),
            UpsampleBlock(z_dim // 2, z_dim // 4),
            UpsampleBlock(z_dim // 4, z_dim // 8),
            UpsampleBlock(z_dim // 8, z_dim // 16),
            UpsampleBlock(z_dim // 16, z_dim // 32),
            nn.Conv2d(z_dim // 32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """

        :param x: shape. = (B, z_dim)
        :return:
        """

        z = self._dense(x).reshape(-1, self.z_dim, *self.map_size)
        return torch.tanh(self._seq(z))


class VAE(nn.Module):

    def __init__(self, z_dim, img_dim, M_N=0.005):
        super().__init__()

        self.M_N = M_N

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim, (img_dim[0] // (2 ** 5), img_dim[1] // (2 ** 5)))

        self.statistic = nn.Linear(z_dim, 2 * z_dim)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        B, C, H, W = x.shape
        encoder_output = self.encoder(x).reshape(B, -1)  # (B, D_Z)
        mu, log_var = self.statistic(encoder_output).chunk(2, dim=-1)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output = self.decoder(z)

        recon_loss = recon(decoder_output, x)
        kl_loss = kl(mu, log_var)

        vae_loss = recon_loss + self.M_N * kl_loss

        return decoder_output, vae_loss


if __name__ == '__main__':
    vae = VAE(128, (64, 64))
    img = torch.rand(2, 3, 64, 64)
    img_recon, vae_loss = vae(img)
    print(img_recon.shape)
    print(vae_loss)
