import torch


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z
