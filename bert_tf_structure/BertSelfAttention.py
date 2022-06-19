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
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.size()
        # check the dimention before hand 
        if len(self.query.weight.size())==3:
            self.query.weight = nn.Parameter(self.query.weight.view(768 , 768))
            self.query.bias = nn.Parameter(self.query.bias.view(768))

            self.key.weight = nn.Parameter(self.key.weight.view(768 , 768))
            self.key.bias = nn.Parameter(self.key.bias.view(768))

            self.value.weight = nn.Parameter(self.value.weight.view(768 , 768))
            self.value.bias = nn.Parameter(self.value.bias.view(768))

        query_layer = self.query(hidden_states)
    
        key_layer = self.key(hidden_states)

        value_layer = self.value(hidden_states)

        query_layer = query_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_len , self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size,seq_len, self.num_attention_heads, self.attention_head_size)

        attention_scores = torch.matmul(key_layer, query_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.view(batch_size , seq_len , hidden_size)
        context_layer = self.dropout(context_layer)
        return context_layer, attention_probs


        

        