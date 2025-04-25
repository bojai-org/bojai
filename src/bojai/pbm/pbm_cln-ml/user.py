from abc import ABC, abstractmethod
import torch
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer
from model import NeuralNetworkCLN


# manages which data user to use depending on task. Used by the deploy stage.
class userManager:
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.tokenizer = tokenizer
        if task_type == "cln":
            self.user = UserCLN(model, tokenizer, device, max_length)


# abstract class for using the model and evaluating it, used as a base for specific data users
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


# the dta user for the GET models.
class UserCLN(User):
    def __init__(self, model, tokenizer, device, max_length):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        output, _ = self.model(input)
        return torch.round(output)
