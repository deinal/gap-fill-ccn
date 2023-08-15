import os
import random
import torch
from torch import nn
import math
from collections import defaultdict


def split_data(data_dir, data_splits):

    assert sum(data_splits) == 1, 'Data splits should add up to one.'

    # Group files by station
    files_by_station = defaultdict(list)
    for f in os.scandir(data_dir):
        if f.is_file() and f.name.endswith('.hdf5'):
            station = f.name.split('_')[0]
            files_by_station[station].append(f.path)

    # Shuffle the files within each station and split according to data_splits
    train_files = []
    val_files = []
    test_files = []
    random.seed(42)
    for station, files in files_by_station.items():
        random.shuffle(files)
        train_ratio, val_ratio = data_splits[:2]
        train_split = int(train_ratio * len(files))
        val_split = train_split + int(val_ratio * len(files))
        train_files.extend(files[:train_split])
        val_files.extend(files[train_split:val_split])
        test_files.extend(files[val_split:])

    return train_files, val_files, test_files


def cyclic_positional_encoding(t, dim, period=1440):
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


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)

    losses = []
    for i, q in enumerate(quantiles):
        errors = target.squeeze() - preds[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))

    return loss


class Time2Vec(nn.Module):
    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0, f'input dim {input_dim} embed dim {embed_dim}'
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            x_output = x
        return x_output
