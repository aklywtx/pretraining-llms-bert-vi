# reference to: https://github.com/rishub-tamirisa/transformer-mlm

import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    """
    Generates positional embeddings using sinusoidal functions to capture positional information.
    """
    def __init__(self, d_model: int, dropout:float=0, max_len:int=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe.require_grad = False
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]
    