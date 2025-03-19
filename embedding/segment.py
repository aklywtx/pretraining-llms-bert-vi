# reference: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py

import torch.nn as nn

class SegmentEmbedding(nn.Embedding):
    """
    Embedding to distinguish different segments (e.g., sentence A and sentence B) in a sequence.
    """

    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)