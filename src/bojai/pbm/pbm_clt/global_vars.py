from model import CLTModelRNN, VocabTokenizer, CharTokenizer
import numpy as np

# file : 1 , dir : 0

browseDict = {
    "train": False,
    "prep": False,
    "deploy_new_data": False,
    "use_model_upload": False,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter text to see output",
    "init": False,
    "type": 2,  # 0 is image, 1 is voice, 2 is text or numbers
    'eval matrice' : 'accuracy',
    "options": 1,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": 0,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}

options = {
    "tokenize using characters (less robust but faster)": CharTokenizer,
    "tokenize using vocabulary (more robust but slower)": VocabTokenizer,
}


def getNewTokenizer():
    pass


def getNewModel():
    return CLTModelRNN()


def init_model(data, model, hypers):
    output_size = len(np.unique(data.processor.outputs))
    model.initialise(
        hypers["hidden_size"], output_size, hypers["num_batches"], hypers["num_layers"]
    )


task_type = "clt"
hyper_params = {
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "num_batches": 32,
    "hidden_size": 256,
    "num_layers": 3,
}
