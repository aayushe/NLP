import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


BERT_URL = 'https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1'
module = hub.Module(BERT_URL)
