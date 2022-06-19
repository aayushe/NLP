import imp
import  torch.nn as nn
from BertSelfAttention import BertSelfAttention
from BertSelfOutput import BertSelfOutput

# calculate output of bert attention and add attetnion output with hidden_states for the feed forward layer 
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        context , attention_output  = self.attention(hidden_states, attention_mask)
        out = self.output(context, hidden_states)
        return out
        