from abc import ABC
from torch import nn
from transformers import AutoModel


# an abstract model used as a base for other models
class Model(ABC):
    def __init__(self):
        super().__init__()


# a model for fine-tuning transformers. Used for GET medium and large
class FineTunedTransformerGET(Model, nn.Module):
    def __init__(self, model_name, vocab_size):
        super(FineTunedTransformerGET, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.lm_head = nn.Linear(self.bert.config.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        return logits
