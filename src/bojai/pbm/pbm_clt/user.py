from abc import ABC, abstractmethod
import torch
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer


# manages which data user to use depending on task. Used by the deploy stage.
class userManager:
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.tokenizer = tokenizer
        if task_type == "clt":
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
        encoding = self.tokenizer.encode(input)
        x = encoding.to(torch.float32).unsqueeze(0)
        x = x.unsqueeze(1).expand(-1, 100, 100)
        x = x.to(torch.float32)
        hidden = self.model.initHidden(1)
        print(hidden.shape)
        output, _ = self.model(x, hidden)
        return torch.argmax(output).item()
