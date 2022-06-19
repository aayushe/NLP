import torch.nn as nn

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        if self.dense.weight.size()[0] == hidden_states.size()[-1]:
            self.dense.weight = nn.Parameter(self.dense.weight.transpose(0,1))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
