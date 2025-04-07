from model import CLIModelCNN
import numpy as np
import requests
# file : 1 , dir : 0

browseDict = {
    'train' : False,
    'prep' : False, 
    'deploy_new_data' : False,
    'use_model_upload' : False, # if 1 means the use_model needs a browse, if not it doesn't.
    'use_model_text' : "Enter a photo to see model output", 
    'init' : False,
    'type' : 0, #0 is image, 1 is voice, 2 is text or numbers
    'eval matrice' : 'accuracy'
}


def getNewTokenizer():
    return None

def getNewModel():
    return CLIModelCNN()

def init_model(data, model):
    dim = data.processor[0][0].shape
    output_size = len(np.unique(data.processor.labels))
    model.initialise(dim, output_size)

task_type = 'cli'

hyper_params = {
    'learning_rate': 1e-5,
    'num_epochs' : 10,
    'num_batches' : 1
    }


class CannotUseFunctionException(Exception):
     pass
