import math
import torch
import pytorch_lightning as pl


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

class LossCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {'train': [], 'val': []}

    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = trainer.callback_metrics['train_loss']
        self.losses['train'].append(avg_train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_val_loss = trainer.callback_metrics['val_loss']
        self.losses['val'].append(avg_val_loss.item())