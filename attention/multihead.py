# reference: https://github.com/rishub-tamirisa/transformer-mlm

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): Embedding dimension size of the input.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability after attention.
        """
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_weights_list = nn.ModuleList()
        
        for _ in range(num_heads):
            q_proj = nn.Linear(embed_dim, self.head_dim)
            k_proj = nn.Linear(embed_dim, self.head_dim)
            v_proj = nn.Linear(embed_dim, self.head_dim)
            self.qkv_weights_list.append(nn.ModuleList([q_proj, k_proj, v_proj]))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        """
        Computes scaled dot-product attention scores and applies attention weights to values.

        Args:
            query (Tensor): Query tensor [batch_size, seq_len, head_dim].
            key (Tensor): Key tensor [batch_size, seq_len, head_dim].
            value (Tensor): Value tensor [batch_size, seq_len, head_dim].
            mask (Tensor, optional): Attention mask to prevent attention to certain positions.

        Returns:
            Tensor: Output of the attention mechanism.
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output
    

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            query (Tensor): Input tensor for queries [batch_size, seq_len, embed_dim].
            key (Tensor): Input tensor used for keys [batch_size, seq_len, embed_dim].
            value (Tensor): Input tensor for values [batch_size, seq_len, embed_dim].
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor after multi-head attention and projection.
        """
        heads_output = []
        for Q, K, V in self.qkv_weights_list:
            q_proj = Q(query)
            k_proj = K(key)
            v_proj = V(value)
            head_output = self.attention(q_proj, k_proj, v_proj, mask)
            heads_output.append(head_output)

        attn_output = torch.cat(heads_output, dim=-1)
        output = self.dropout(self.out_proj(attn_output))
        return output
