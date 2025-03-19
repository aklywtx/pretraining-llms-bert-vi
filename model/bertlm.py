# reference: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/language_model.py

import torch.nn as nn
from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Masked Language Modelling + Sentence Order Prediction/Next Sentence Prediction Model
    """

    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm_head = MaskedLanguageModel(bert.embed_dim, vocab_size)
        self.sop_head = SentenceOrderPredictionHead(bert.embed_dim)

    def forward(self, input_ids, token_type_ids, input_mask):
        sequence_output = self.bert(input_ids, token_type_ids, input_mask)
        # project bert output to logits
        mlm_logits = self.mask_lm_head(sequence_output)
        sop_logits = self.sop_head(sequence_output)
        return mlm_logits, sop_logits


class MaskedLanguageModel(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states)
    
class SentenceOrderPredictionHead(nn.Module):
    """
    2-class classification model: correct order, incorrect order
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, hidden_states):
        cls_token_embedding = hidden_states[:, 0, :]
        return self.linear(cls_token_embedding)