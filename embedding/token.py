# refer to https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/token.py

import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Embedding layer converting token IDs into dense embedding vectors.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim, padding_idx=0)