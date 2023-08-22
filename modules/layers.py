import torch
import torch.nn as nn
import torch.nn.functional as F


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
