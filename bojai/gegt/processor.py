from abc import ABC, abstractmethod
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import requests
import torch

#manages which data processor to use, used by the prep stage
class ProcessorManager():
    def __init__(self, data_dir, division, model, device, tokenizer, task_type):
        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.processor = None
        self.decide_which_processor(task_type)
    
    def decide_which_processor(self, task_type):
        if task_type == 'gegt':
            self.processor = ProcessorImage2Text(self.data_dir, self.division, self.model, self.device, self.tokenizer)
            self.train =  ProcessorImage2Text(self.processor.train_dir, [0,1], self.model, self.device, self.tokenizer, False)
            self.eval =  ProcessorImage2Text(self.processor.eval_dir, [1,0], self.model, self.device, self.tokenizer, False)
    


#abstract class serving as the base for the other types of processors. 
#The classes extending it should: 
# - should split data
# - be able to accept some ops on data if they process numbers
# - should tokenize data
# - should shuffle data
# - should handle missing data and duplicates
class Processor(ABC):
    def __init__(self, data_dir, division, model, device, tokenizer):
        super().__init__()
        
        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.inputs_train, self.inputs_eval, self.outputs_train, self.outputs_eval = None, None, None,None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_item_untokenized(self,idx):
        pass


class ProcessorImage2Text(Processor, Dataset):
    def __init__(self, data_dir, division, model, device, tokenizer, main = True):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.images, self.texts = self.get_inputs_outputs()
        train, eval = division
        if train + eval != 1: 
            raise ValueError("the division must add to 1")
        
        self.image_processor = self.tokenizer[0]
        self.tokenizer_text = self.tokenizer[1]
        if main: 
            self.inputs_train, self.inputs_eval, self.outputs_train, self.outputs_eval = self.divide()
            self.train_dir, self.eval_dir = self.get_eval_train()
    
    def get_inputs_outputs(self):
        images = []
        image_dir = os.path.join(self.data_dir, 'input')
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                img_path = os.path.join(image_dir, filename) 
                img = Image.open(img_path)  
                images.append(img)  
            else: 
                raise ValueError("All data in input directory must be images")

        outputs_file = os.path.join(self.data_dir, 'output.txt')
        with open(outputs_file, 'r', encoding='utf-8') as f:
            outputs = f.readlines()
        outputs = [line.strip() for line in outputs]

        return images, outputs   
    def divide(self):
        # Determine the indices for splitting
        num_samples = len(self.texts)
        train_size = int(num_samples * self.division[0])  # Training set size
        indices = np.random.permutation(num_samples)  # Randomly shuffle the indices
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        # Split inputs and outputs
        inputs_train = [self.images[i] for i in train_indices]
        inputs_eval = [self.images[i] for i in eval_indices]
        outputs_train = [self.texts[i] for i in train_indices]
        outputs_eval = [self.texts[i] for i in eval_indices]

        return inputs_train, inputs_eval, outputs_train, outputs_eval

    def __getitem__(self, idx):
      image_features = self.image_processor(self.images[idx], return_tensors="pt").pixel_values
      labels = self.tokenizer_text(self.texts[idx],return_tensors="pt",
                                          max_length=46,
                                          pad_to_max_length=True,
                                          return_token_type_ids=True,
                                          truncation=True).input_ids
      labels = labels.squeeze(0)

      return image_features.squeeze(0), labels

    def __len__(self):
        return len(self.texts)
    
        #returns the divided data, for evalaution and training
    def get_eval_train(self):
        os.makedirs('train_dir', exist_ok=True)
        os.makedirs('eval_dir', exist_ok=True)

        # Save train data to text files
        self.save_to_file_image(os.path.join('train_dir', "input"), self.inputs_train)
        self.save_to_file(os.path.join('train_dir', "output.txt"), self.outputs_train)

        # Save evaluation data to text files
        self.save_to_file_image(os.path.join('eval_dir', "input"), self.inputs_eval)
        self.save_to_file(os.path.join('eval_dir', "output.txt"), self.outputs_eval)

        return 'train_dir', 'eval_dir'
    
    def get_item_untokenized(self,idx):
        return self.images[idx], self.texts[idx]
    
    #Helper function to write a list to a file, one sentence per line.
    def save_to_file(self, file_path, data_list):
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence in data_list:
                f.write(sentence.strip() + "\n")
    
    def save_to_file_image(self, file_path, data_list):
        if not os.path.exists(file_path):
            os.makedirs(file_path)  # Create directory if it doesn't exist

        for i, img in enumerate(data_list, start=1):
            dir_path = os.path.join(file_path, f"{i}.jpg")  # Save as .jpg
            img.save(dir_path, format="JPEG")  # You can change format to PNG if needed