import tensorflow as tf
from Bert import BERT
import tensorflow as tf

import numpy as np
import json
import torch
from Bert import BERT
from BertConfig import BertConfig
path = "./config.json"
config = BertConfig(path)
bert = BERT(config)

torch_list = [
"embeddings.word_embeddings.weight",
"embeddings.position_embeddings.weight",
"embeddings.token_type_embeddings.weight",
"embeddings.LayerNorm.weight",
"embeddings.LayerNorm.bias",
"encoder.layers.0.attention.attention.query.weight",
"encoder.layers.0.attention.attention.query.bias",
"encoder.layers.0.attention.attention.key.weight",
"encoder.layers.0.attention.attention.key.bias",
"encoder.layers.0.attention.attention.value.weight",
"encoder.layers.0.attention.attention.value.bias",
"encoder.layers.0.attention.output.dense.weight",
"encoder.layers.0.attention.output.dense.bias",
"encoder.layers.0.attention.output.LayerNorm.weight",
"encoder.layers.0.attention.output.LayerNorm.bias",
"encoder.layers.0.intermediate.dense.weight",
"encoder.layers.0.intermediate.dense.bias",
"encoder.layers.0.output.dense.weight",
"encoder.layers.0.output.dense.bias",
"encoder.layers.0.output.LayerNorm.weight",
"encoder.layers.0.output.LayerNorm.bias",
'pooler.dense.weight',
'pooler.dense.bias'
]

tf_list = [ 'word_embeddings/embeddings:0',
    'position_embedding/embeddings:0',
    'type_embeddings/embeddings:0',
    'embeddings/layer_norm/gamma:0',
    'embeddings/layer_norm/beta:0',
    'transformer/layer_0/self_attention/query/kernel:0',
    'transformer/layer_0/self_attention/query/bias:0',
    'transformer/layer_0/self_attention/key/kernel:0',
    'transformer/layer_0/self_attention/key/bias:0',
    'transformer/layer_0/self_attention/value/kernel:0',
    'transformer/layer_0/self_attention/value/bias:0',
    'transformer/layer_0/self_attention/attention_output/kernel:0',
    'transformer/layer_0/self_attention/attention_output/bias:0',
    'transformer/layer_0/self_attention_layer_norm/gamma:0',
    'transformer/layer_0/self_attention_layer_norm/beta:0',
    'transformer/layer_0/intermediate/kernel:0',
    'transformer/layer_0/intermediate/bias:0',
    'transformer/layer_0/output/kernel:0',
    'transformer/layer_0/output/bias:0',
    'transformer/layer_0/output_layer_norm/gamma:0',
    'transformer/layer_0/output_layer_norm/beta:0',
    'pooler_transform/kernel:0', 
    'pooler_transform/bias:0']


def map_tf_torch_params():
    torch_tf_mapping = dict(zip(torch_list, tf_list))
    layer_dict = {}
    for i in range(1, 12):
        for key, value in torch_tf_mapping.items():
            if "layers" in key:
                new_key = key.replace("layers.0", f"layers.{i}")
                new_value = value.replace("layer_0", f"layer_{i}")
                layer_dict[new_key] = new_value
        torch_tf_mapping.update(layer_dict)

    return torch_tf_mapping

################################# TF 2 , load pb file and get tensors ##################
def get_tf_weights():
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    export_dir = "model"
    model = tf.saved_model.load(
        export_dir, tags=None
    )
    trainable_variables = model.trainable_variables
    trainable_variables_tf = {}
    for i, each in enumerate(trainable_variables):
        key = each.name
        trainable_variables_tf[key] = each.numpy()
    return trainable_variables_tf

def load_params_on_bert():
    trainable_variables_tf_dict = get_tf_weights()
    mapping = map_tf_torch_params()
    weighted_params = {}
    for torch_name, tf in mapping.items():
        weighted_params[torch_name] = trainable_variables_tf_dict[tf]

    for name, param in bert.named_parameters():
        value = weighted_params[name]
        param.data = torch.from_numpy(np.asarray(value))
    return bert