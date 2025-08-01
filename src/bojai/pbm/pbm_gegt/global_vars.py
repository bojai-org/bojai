from model import FineTunedTransformerGEGT
from transformers import ViTImageProcessor, BertTokenizer

# file : 1 , dir : 0
browseDict = {
    "train": False,
    "prep": False,
    "deploy_new_data": False,
    "use_model_upload": True,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter one picture to see output",
    "init": False,
    "type": 0,  # 0 is image, 1 is voice, 2 is text or numbers
    "eval matrice": "perplexity",
    "options": 0,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": -1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    image_processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")
    return [image_processor, tokenizer_text]


def getNewModel():
    return FineTunedTransformerGEGT(len(getNewTokenizer()[1]))


def init_model(data, model, hyper_params):
    pass


task_type = "gegt"
hyper_params = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "num_workers": 0,
}


class CannotUseFunctionException(Exception):
    pass


options = {}
