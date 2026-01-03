from model import LogisticRegressionCLN
import numpy as np

# file : 1 , dir : 0

browseDict = {
    "train": True,
    "prep": True,
    "deploy_new_data": True,
    "use_model_upload": False,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter comma-separated numbers",
    "init": True,
    "type": 2,  # 0 is image, 1 is voice, 2 is text or numbers
    "eval matrice": "accuracy",
    "options": 0,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": -1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    return None


def getNewModel():
    return LogisticRegressionCLN()


def init_model(data, model, not_used=None):
    n, d = data.processor.inputs.shape
    model.initialise(d)


task_type = "cln"
hyper_params = {
    "learning_rate": 1e-5,
}
options = {}
