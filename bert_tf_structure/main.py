
from doctest import OutputChecker
from Bert import BERT
from BertConfig import BertConfig


if __name__ == '__main__':
    # load the model
    # path = "./config.json"
    # config = BertConfig(path)
    # model = BERT(config=config)
    from convert_tf_to_torch import *

    def tokenize(text):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        return inputs

    text = [
            'Hello, how are you? I am Romeo.\n',
            'Hello, Romeo My name is Juliet. Nice to meet you.\n',
            'Nice meet you too. How are you today?\n',
            'Great. My baseball team won the competition.\n',
            'Oh Congratulations, Juliet\n',
            'Thanks you Romeo'
        ]

    inputs = tokenize(text)

    input_ids = inputs["input_ids"]
    segment_ids = inputs["token_type_ids"]
    masked_pos = inputs["attention_mask"]

    model = load_params_on_bert()

    output_embedding , pooled_out = model(input_ids=input_ids , segment_ids=segment_ids , attention_mask=None)

        
    # output from custom bert model 


    # output from torch bert model 
