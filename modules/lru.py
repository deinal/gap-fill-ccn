import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class LRUCell(nn.Module):
    """
    Parallel LRU PyTorch implementation
    https://github.com/TingdiRen/LRU_pytorch
    """
    def __init__(self, in_features, activation=torch.relu, dropout=0.0, 
                 r_min=0.9, r_max=0.999, use_bias=True, unroll=False):
        super(LRUCell, self).__init__()
        self.hidden_size = in_features
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.unroll = unroll  # The parallel algorithm will divide and conquer more if True

        self.i_dense = nn.Linear(in_features, in_features * 2, bias=use_bias)  # Extend to the complex C
        self.o_dense = nn.Linear(in_features * 2, in_features, bias=use_bias)  # Back to real R

        # Initialize parameters
        u1 = np.random.random(size=in_features)
        u2 = np.random.random(size=in_features)
        v_log = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))  # defined in [arxiv] lemma 3.2
        theta_log = np.log(u2 * np.pi * 2)  # defined in [arxiv] lemma 3.2
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(v_log)) ** 2))  # defined above eq.7 of [arxiv]

        # defined in Optimization under exponential parameterization of [arxiv] 3.3
        self.params_log = nn.Parameter(torch.tensor([v_log, theta_log, gamma_log], dtype=torch.float32))

    def lru_parallel(self, i, x, v, theta, B, L, D):
        # Upper/low parallel algorithm
        l = 2 ** i
        x = x.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        x1, x2 = x[:, :l // 2], x[:, l // 2:]  # Divide the data in half

        pos = torch.arange(1, l // 2 + 1, dtype=torch.float, device=x.device)  # t=k+1 ~ T
        vs = torch.einsum('n,d->nd', pos, v)
        thetas = torch.einsum('n,d->nd', pos, theta)
        lambs = torch.exp(
            torch.complex(-vs, thetas))  # defined in Optimization under exponential parameterization of [arxiv] 3.3

        x2 = x2 + (lambs * x1[:, -1:])  # Add the last element of the half to the second half
        x = torch.cat([x1, x2], axis=1)
        if (not self.unroll) and x.shape[1] is not None:
            x = x.reshape(B, L, D)

        return i + 1, x, v, theta, B, L, D

    def while_loop(self, cond, body, loop_vars):
        while cond(*loop_vars[:2]):
            loop_vars = body(*loop_vars)
        return loop_vars

    def forward(self, inputs):
        u = self.i_dense(inputs)
        params = torch.exp(self.params_log)
        v, theta, gamma = params[0], params[1], params[2]

        len_seq_in = u.size(1)
        log2_L = int(np.ceil(np.log2(len_seq_in)))

        u = torch.view_as_complex(u.view(u.size(0), u.size(1), u.size(2) // 2, 2))
        u = F.pad(u,
                  (0, 0, 0, 2 ** log2_L - u.size(1), 0, 0))  # pad the sequence length to the power of 2 (for algorithm)
        B, L, D = u.size(0), u.size(1), u.size(2)

        if self.unroll:
            x = u  # init hidden states as inputs
            for i in range(log2_L):
                _, x, *_ = self.lru_parallel(i + 1, x, v, theta, B, L, D)
        else:
            _, x, *_ = self.while_loop(lambda i, x: i <= log2_L, self.lru_parallel, [1, u, v, theta, B, L, D])

        x = x[:, :len_seq_in] * (gamma.to(torch.complex64) + 0j)  # Element-wise parameter defined in [arxiv] eq.(7)
        x = self.complex_to_real_imag(x)
        output = self.o_dense(x)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output

    def complex_to_real_imag(self, x):
        real_x = torch.real(x)
        imag_x = torch.imag(x)
        return torch.cat((real_x, imag_x), dim=-1)


class BidirectionalLRUCell(nn.Module):
    def __init__(self, in_features, dropout):
        super().__init__()
        self.forward_cell = LRUCell(in_features, dropout=dropout)
        self.backward_cell = LRUCell(in_features, dropout=dropout)

    def forward(self, input):
        # Forward pass
        forward_output = self.forward_cell(input)

        # Backward pass
        reversed_input = torch.flip(input, dims=[1])
        backward_output = self.backward_cell(reversed_input)
        backward_output = torch.flip(backward_output, dims=[1])

        # Concatenate forward and backward outputs
        output = torch.cat([forward_output, backward_output], dim=-1)
        return output


class LRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if bidirectional:
            output_size = 2*hidden_size
        else:
            output_size = hidden_size

        layers = []

        layers.append(nn.Linear(input_size, hidden_size))

        for i in range(num_layers):
            if bidirectional:
                layers.append(
                    BidirectionalLRUCell(hidden_size, dropout))    
            else:
                layers.append(LRUCell(hidden_size, dropout))
            
            # Interleave with MLP
            if i < num_layers - 1:
                layers.append(nn.Sequential(
                    nn.Linear(output_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))

        self.layers = nn.ModuleList(layers)

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)

        return x
