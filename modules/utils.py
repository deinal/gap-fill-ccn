import math
import torch


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

def scaled_dot_product_attention(self, query, key, value):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_attention_logits = matmul_qk / self.scale
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output