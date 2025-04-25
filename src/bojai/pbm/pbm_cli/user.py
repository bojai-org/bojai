from abc import ABC, abstractmethod
import torch
from PIL import Image
import numpy as np


#manages which data user to use depending on task. Used by the deploy stage. 
class userManager():
    def __init__(self, task_type, model, tokenizer, device, max_length = None):
        self.user = None
        self.tokenizer = tokenizer
        if task_type == 'cli':
            self.user = UserCLI(model, tokenizer, device, max_length)
        

#abstract class for using the model and evaluating it, used as a base for specific data users
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


#the dta user for the GET models. 
class UserCLI(User):
    def __init__(self, model , tokenizer, device, max_length):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        img = Image.open(input).convert("RGB")
        img = img.resize((500, 500))
        img_array = np.array(img)
        
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).float() / 255.0  # [3, 500, 500]
        x = img_tensor.unsqueeze(0)  # [1, 3, 500, 500]
        
        with torch.no_grad():  # optional, for inference
            y_predicted = self.model(x)
            pred = torch.argmax(y_predicted, dim=1)

        return pred.item()  # return int instead of tensor
