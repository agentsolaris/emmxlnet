import os

from pytorch_transformers.modeling_xlnet import XLNetConfig,XLNetModel
from torch import nn


class BertModule(nn.Module):
    def __init__(self, bert_model_name, dropout_prob=0.1, cache_dir="./cache/"):
        super().__init__()

        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = XLNetModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir, output_hidden_states=True
        )
        self.bert_model.train()

    def forward(self, token_ids, token_segments,token_type_ids=None, attention_mask=None):
        loss,  pooled_output, encoded_layers = self.bert_model(
            token_ids, token_type_ids=None
        )
        return encoded_layers, pooled_output
