import torch
import torch.nn as nn
import numpy as np

# creating embedding for input ids means to project the input to the embeddding space.
# the embedding space is a vector space.
# there are three embedding space we can project the input in 
# 1. Word Embedding
# 2. Positional Embedding
# 4. Token type Embedding


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_pos_embed, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.segment_len, config.hidden_size)
        self.max_position_embeddings = config.max_pos_embed
        self.hidden_size = config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout, inplace=False)

    def get_positions(self):
        # create sinosoidal position embedding for each dim  
        self.all_pos = np.empty((self.max_position_embeddings, self.hidden_size))
        for pos in range(self.max_position_embeddings):
            for i in range(self.hidden_size):
                if i % 2 == 0:
                    np.append(self.all_pos, np.sin(pos / np.power(10000, 2 * i / self.hidden_size)))
                else:
                    np.append(self.all_pos, np.cos(pos / np.power(10000, (2 * i + 1) / self.hidden_size)))
        all_pos = torch.as_tensor(self.all_pos, dtype=torch.long)
            
        return all_pos

    def forward(self, input_ids, token_type_ids=None):
        numeric = True
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]
        
        # create embedding for input ids
        word_embeddings = self.word_embeddings(input_ids)
        if numeric:
            numeric_pos = torch.arange(self.max_position_embeddings)[:seq_len]
            print(numeric_pos.shape)
            numeric_pos = numeric_pos.expand(batch_size,-1)
            position_embeddings = self.position_embeddings(numeric_pos)
        else:
            all_pos = self.get_positions()
            input_pos =all_pos[:seq_len,:]
            input_pos = input_pos.expand(batch_size , -1 , -1)
            position_embeddings = input_pos
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        print(word_embeddings.shape , position_embeddings.shape , token_type_embeddings.shape)
        
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings