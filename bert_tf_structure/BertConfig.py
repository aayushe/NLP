import json

class BertConfig():
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.heads = self.config["heads"]
        self.dropout = self.config["dropout"]
        self.max_pos_embed = self.config["max_pos_embed"]
        self.segment_len = self.config["segment_len"]
        self.intermediate_size = self.config["intermediate_size"]
        self.layer_norm_eps = self.config["layer_norm_eps"]
        self.hidden_act = self.config["hidden_act"]

    def __str__(self):
        return str(self.config)
        

