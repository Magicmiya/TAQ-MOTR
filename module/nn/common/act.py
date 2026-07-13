# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import torch
import torch.nn as nn


def get_activation(act: str | None, inplace: bool = True):
    """
    Get activation module
    Args:
        act: act module name
        inplace: inplace flag

    Returns:
        activation module
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == 'silu' or act == 'swish':
        m = nn.SiLU(inplace=inplace)

    elif act == 'relu':
        m = nn.ReLU(inplace=inplace)

    elif act == 'leaky_relu':
        m = nn.LeakyReLU(inplace=inplace)

    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid(inplace=inplace)

    else:
        raise RuntimeError('')

    return m


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
