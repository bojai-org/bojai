from model import FineTunedTransformerGET
from transformers import AutoTokenizer, AutoModel
import numpy as np

# file : 1 , dir : 0

browseDict = {
    "train": False,
    "prep": False,
    "deploy_new_data": False,
    "use_model_upload": False,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter a sentence to see the model output.",
    "init": False,
    "type": 2,  # 0 is image, 1 is voice, 2 is text or numbers
    "eval matrice": "perplexity",
    "options": 0,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": -1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    return AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


def getNewModel():
    return FineTunedTransformerGET(
        "huawei-noah/TinyBERT_General_4L_312D", vocab_size=len(getNewTokenizer())
    )


def init_model(data, model, hyper_params):
    pass


task_type = "get"
hyper_params = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "num_workers": 0,
}
options = {}
