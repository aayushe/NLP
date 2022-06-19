# contains multihead attetnion and feed forward layers
from BertLayer import BertLayer
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [
                BertLayer(config)
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, x , attention_mask):
        for layer in self.layers:
            x = layer(x,attention_mask)
        print("x" , x.shape)
        return x