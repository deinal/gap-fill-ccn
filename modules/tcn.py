import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size-1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=self.padding , dilation=dilation)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x += residual

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

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