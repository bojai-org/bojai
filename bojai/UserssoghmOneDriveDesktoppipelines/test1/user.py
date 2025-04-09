from abc import ABC, abstractmethod
import torch
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer
from model import FineTunedTransformerGEGT
#manages which data user to use depending on task. Used by the deploy stage. 
class userManager():
    def __init__(self, task_type, model, tokenizer, device, max_length = None):
        self.user = None
        self.tokenizer = tokenizer
        if task_type == '@TODO enter-model-name':
            self.user = ImplementYourUser(model, tokenizer, device, max_length)
        

#abstract class for using the model, used as a base for specific data users
class User(ABC):
    def __init__(self, model, tokenizer, device, max_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    @abstractmethod
    def use_model(self, input):
        pass

'''
@TODO 
Implement your own model user, it must extend User and implement use_model. 

use_model is fuction that takes an input and runs it into the model to get an output. 

'''
class ImplementYourUser(User):
    pass