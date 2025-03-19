import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    Combines multiple embedding layers used in BERT:
        1. TokenEmbedding: Maps token indices to dense embeddings.
        2. PositionalEmbedding: Adds positional information using sinusoidal functions.
        3. SegmentEmbedding: Indicates the sentence segment (sentence A or sentence B).

    The sum of these embeddings, followed by dropout and layer normalization, forms the output.
    """

    def __init__(self, vocab_size, embed_size, dropout=0.0):
        """
        Initializes the BERTEmbedding module.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embeddings.
            dropout (float): Dropout probability applied after embeddings.
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.token.embedding_dim)
        self.embed_size = embed_size

    def forward(self, input_ids, token_type_ids):
        """
        Computes the combined embeddings.

        Args:
            input_ids (Tensor): Token IDs of input sequences [batch_size, seq_len].
            token_type_ids (Tensor): Segment token IDs indicating sentence A or B [batch_size, seq_len].

        Returns:
            Tensor: Final embeddings after summation, dropout, and layer normalization.
        """
        embeddings = (self.token(input_ids) + self.position(input_ids) + self.segment(token_type_ids))
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings