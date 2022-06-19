# attention is a function that takes keys , queries , values and does the attention calculation:
# attention = softmax(QK^T / (d_k**0.5)) where d_k is the dimension of the key

import torch.nn as nn
import math
import torch

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.heads
        self.attention_head_size = int(config.hidden_size / config.heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # implemented according to original paper
        self.query = nn.Linear(self.attention_head_size,self.attention_head_size)
        self.key = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.value = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.size()
        print("HERE " , hidden_states.shape)
        # query_layer = self.query(hidden_states)
        # key_layer = self.key(hidden_states)
        # value_layer = self.value(hidden_states)

        query = hidden_states.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        key = hidden_states.view(batch_size, seq_len , self.num_attention_heads, self.attention_head_size)
        value = hidden_states.view(batch_size,seq_len, self.num_attention_heads, self.attention_head_size)
        print(query.shape , self.query)
        query_layer = self.query(query)
        key_layer = self.key(key)
        value_layer = self.value(value)

        print("query_layer" , query_layer.shape)
        attention_scores = torch.matmul(key_layer, query_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.view(batch_size , seq_len , hidden_size)
        context_layer = self.dropout(context_layer)
        return context_layer, attention_probs


        

        