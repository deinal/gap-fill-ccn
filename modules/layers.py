import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.l0 = nn.Linear(in_features, 1)
        self.li = nn.Linear(in_features, out_features - 1)
        self.f = torch.sin

    def forward(self, tau):
        time_linear = self.l0(tau) # ω0 * τ + φ0
        time_sin = self.f(self.li(tau)) # f(ωi * τ + φi)
        encoded_time = torch.cat([time_linear, time_sin], -1)  
        return encoded_time


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_input, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.wq = nn.Linear(d_input, d_model)
        self.wk = nn.Linear(d_input, d_model)
        self.wv = nn.Linear(d_input, d_model)
        self.scale = d_model**0.5
        
    def forward(self, query, key, value):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / self.scale
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
