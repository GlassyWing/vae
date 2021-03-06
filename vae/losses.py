import torch
import numpy as np
from torch.nn import functional as F

def recon(output, target):
    """
    重构损失，这里将 q(x|z) 看作正态分布
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    loss = F.mse_loss(output, target)
    return loss


def kl(mu, log_var):
    """

    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    return torch.mean(loss, dim=0)
