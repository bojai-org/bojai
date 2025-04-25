from model import (
    LogisticRegressionCLN,
    NeuralNetworkCLN,
    DeepNeuralNetworkCLN,
    NeuralNetworkCLNL2,
    NeuralNetworkCLNL1,
    NeuralNetworkCLNElasticNet,
    NeuralNetworkCLNDropout,
    kNN,
)
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
    "use_model_number": True,
    "eval matrice": "accuracy",
    "options": 1,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": 1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    return None


def getNewModel():
    return LogisticRegressionCLN()


def init_model(data, model):
    if model.__class__.__name__ != "kNN":
        n, d = data.processor.input_matrix.shape
        model.initialise(d)
    else:
        model.initialise(data.processor.input_matrix, data.processor.output)


task_type = "cln"
hyper_params = {
    "learning_rate": 1e-5,
}

options = {
    "logistic regression": LogisticRegressionCLN,
    "one-layer neural network": NeuralNetworkCLN,
    "two-layer neural network": DeepNeuralNetworkCLN,
    "one-layer neural network with L2 normalization": NeuralNetworkCLNL2,
    "one-layer neural network with L1 normalization": NeuralNetworkCLNL1,
    "one-layer neural network with L2 and L1 normalization": NeuralNetworkCLNElasticNet,
    "one-layer neural network with dropout": NeuralNetworkCLNDropout,
    "k nearest neighbors": kNN,
}
