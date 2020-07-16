import torch
import torch.nn as nn

from vae.decoder import Decoder
from vae.encoder import Encoder
from vae.losses import recon, kl
from vae.utils import reparameterize


class VAE(nn.Module):

    def __init__(self, z_dim, img_dim, M_N=0.005):
        super().__init__()

        self.M_N = M_N

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        recon_loss = recon(decoder_output, x)
        kl_loss = kl(mu, log_var)
        kl_2_loss = 0.
        mul = 2.
        for ls in losses:
            kl_2_loss += self.M_N * 1 / mul * ls
            mul *= 2.

        vae_loss = recon_loss + self.M_N * kl_loss + kl_2_loss

        return decoder_output, vae_loss


if __name__ == '__main__':
    vae = VAE(512, (64, 64))
    img = torch.rand(2, 3, 64, 64)
    img_recon, vae_loss = vae(img)
    print(img_recon.shape)
    print(vae_loss)
