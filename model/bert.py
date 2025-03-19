# reference: https://github.com/rishub-tamirisa/transformer-mlm

import torch.nn as nn
from .transformer import TransformerBlock
from ..embedding import BERTEmbedding
    
class BERT(nn.Module):
    """
    Implementation of the BERT model consisting of:
        - BERTEmbedding: Combines token, positional, and segment embeddings.
        - Multiple stacked Transformer blocks.

    The final output provides contextualized embeddings for downstream tasks.
    """
    def __init__(self, vocab_size, embed_dim, n_layers, num_heads, dropout=0., ff_dropout=0.):
        """
        Initializes the BERT model.

        Args:
            vocab_size (int): Vocabulary size.
            embed_dim (int): Embedding dimension.
            n_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads per Transformer layer.
            dropout (float): Dropout rate applied after attention layers.
            ff_dropout (float): Dropout rate applied within feed-forward layers.
        """
        super(BERT, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.attn_heads = num_heads
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_dim) # embedding for BERT, sum of positional, segment, token embeddings
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, ff_dropout=ff_dropout) 
            for _ in range(n_layers)]
            ) # multi-layers transformer blocks

        
    def forward(self, input_ids, token_type_ids, input_mask):
        x = self.embedding(input_ids, token_type_ids) # input embedding
        for transformer in self.transformer_blocks: # running over multiple transformer blocks
            x = transformer.forward(x, input_mask)
        return x
    

        