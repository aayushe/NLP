import torch.nn as nn
from Encoder import Encoder
from BertEmbedding import BertEmbedding
import torch
import json
from Pooler import Pooler

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.embeddings = BertEmbedding(config)
        self.encoder = Encoder(config)
        self.pooler = Pooler(config)

    def forward(self, input_ids, segment_ids , attention_mask):
        print("input_ids" , input_ids.shape , segment_ids.shape)
        output = self.embeddings(input_ids, segment_ids)
        embeddings = self.encoder(output ,attention_mask)
        print("embeddings out " , embeddings.shape , embeddings[0].shape)
        pooled_output = self.pooler(embeddings[:1,:1,:])
        print("pooled out " , pooled_output.shape)
        return (embeddings ,  pooled_output+embeddings[1:])

        # return embeddings[1:] , pooled_output