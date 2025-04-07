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
        img = Image.open(input)  
        img = img.resize((500, 500))
        img_array = np.array(img)
        # Convert the NumPy array to a PyTorch tensor
        # The shape of the image_array will be (H, W, 3), so we need to transpose it to (3, H, W)
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()  # Convert to float for neural network processing

        # Normalize the tensor (optional)
        x = img_tensor / 255.0  # Scaling the pixel values to the range [0, 1]
        x = x.to(torch.float32)
        x = x.unsqueeze(0)
        y_predicted = self.model(x)
        pred = torch.argmax(y_predicted, dim=1)
        return pred