import math
import torch


def periodic_positional_encoding(t, dim, period=24):
    assert dim % 2 == 0, 'Dimension must be even.'
    div_term = torch.exp(torch.arange(0., dim, 2) * -(2 * math.log(10000.0) / dim)).to(t.device)
    pe = torch.zeros(t.size(0), t.size(1), dim).to(t.device)
    pe[:, :, 0::2] = torch.sin(2 * math.pi * t / period * div_term)
    pe[:, :, 1::2] = torch.cos(2 * math.pi * t / period * div_term)
    return pe

def baseline_positional_encoding(t, dim):
    assert dim % 2 == 0, 'Dimension must be even.'
    div_term = torch.exp(torch.arange(0., dim, 2) * -(2 * math.log(10000.0) / dim)).to(t.device)
    pe = torch.zeros(t.size(0), t.size(1), dim).to(t.device)
    pe[:, :, 0::2] = torch.sin(t * div_term)
    pe[:, :, 1::2] = torch.cos(t * div_term)
    return pe
