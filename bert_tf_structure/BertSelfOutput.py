import torch.nn as nn

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        print("####" , hidden_states.shape , self.dense.weight.shape)
        if self.dense.weight.size()[-1] == hidden_states.size()[-1]:
            self.dense.weight  = nn.Parameter(self.dense.weight.view(-1,768))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states