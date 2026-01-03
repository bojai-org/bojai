from model import CLIModelCNN
import numpy as np

# file : 1 , dir : 0

browseDict = {
    "train": False,
    "prep": False,
    "deploy_new_data": False,
    "use_model_upload": True,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter a photo to see model output",
    "init": False,
    "type": 0,  # 0 is image, 1 is voice, 2 is text or numbers. Used to view data in appropriate ways
    "eval matrice": "accuracy",
    "options": 0,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": -1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    return None


def getNewModel():
    return CLIModelCNN()


def init_model(data, model, hyper_params):
    dim = data.processor[0][0].shape
    output_size = len(np.unique(data.processor.labels))
    model.initialise(dim, output_size)


task_type = "cli"

hyper_params = {"learning_rate": 1e-5, "num_epochs": 10, "num_batches": 1}


class CannotUseFunctionException(Exception):
    pass


options = {}
