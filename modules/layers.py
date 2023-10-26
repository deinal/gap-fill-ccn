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


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size-1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=self.padding , dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)   
        x = x.transpose(1, 2)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x += residual
        x = self.gelu(x)
        return x


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        channels = [hidden_size] * n_layers
        for i in range(n_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else channels[i-1]
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (Batch, Length, Channels)
        x = self.network(x.transpose(1, 2)) # (Batch, Channels, Length)
        x = x.transpose(1, 2) # (Batch, Length, Channels)
        return x
