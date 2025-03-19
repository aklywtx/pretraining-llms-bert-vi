# reference to: https://github.com/rishub-tamirisa/transformer-mlm

import torch.nn as nn
from ..attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of a Multi-Head Attention mechanism followed by a Feed-Forward network,
    each with residual connections and layer normalization.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., ff_dropout=0.):
        """
        Initializes the TransformerBlock module.

        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability applied after attention.
            ff_dropout (float): Dropout probability in the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward =FeedForward(embed_dim=embed_dim, ff_dropout=ff_dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        attn_output = self.multi_head_attention(x, x, x, mask=mask)
        x = self.layer_norm1(attn_output + x)
        
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(ff_output + x)
        return x
        
class FeedForward(nn.Module):
    """
    Feed-forward neural network component used within Transformer blocks.
    """
    def __init__(self, embed_dim, width_fac=4, ff_dropout=0.):
        """
        Initializes the FeedForward module.

        Args:
            embed_dim (int): Embedding dimension of the input.
            width_fac (int): Expansion factor for the hidden layer dimension.
            ff_dropout (float): Dropout rate applied after activation.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, width_fac * embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(ff_dropout)
        self.linear2 = nn.Linear(width_fac * embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x