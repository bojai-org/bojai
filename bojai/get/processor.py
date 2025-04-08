from abc import ABC, abstractmethod
import torch
import os
from torch.utils.data import Dataset
import numpy as np

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
        if task_type == 'get': 
            self.processor = ProcessorSeq2Seq(self.data_dir, self.division, self.model, self.device, self.tokenizer,512)
            self.train = ProcessorSeq2Seq(self.processor.train_dir, [1,0], self.model, self.device, self.tokenizer,512, False)
            self.eval = ProcessorSeq2Seq(self.processor.eval_dir, [0,1], self.model, self.device, self.tokenizer,512, False)
    


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

#processes data for sequence to sequence models. 
#the file must be stored in two txt files, one for input and one for output. 
class ProcessorSeq2Seq(Processor, Dataset):
    def __init__(self, data_dir, division, model, device, tokenizer, max_length, main = True):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.inputs, self.outputs, self.combined = self.get_inputs_outputs()
        train, eval = division
        self.max_length = max_length
        if train + eval != 1: 
            raise ValueError("the division must add to 1")
        if main:
            self.inputs_train, self.inputs_eval, self.outputs_train, self.outputs_eval = self.divide()
            self.train_dir, self.eval_dir = self.get_eval_train()
            

    def divide(self):
        # Determine the indices for splitting
        num_samples = len(self.inputs)
        train_size = int(num_samples * self.division[0])  # Training set size
        indices = np.random.permutation(num_samples)  # Randomly shuffle the indices
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        # Split inputs and outputs
        inputs_train = [self.inputs[i] for i in train_indices]
        inputs_eval = [self.inputs[i] for i in eval_indices]
        outputs_train = [self.outputs[i] for i in train_indices]
        outputs_eval = [self.outputs[i] for i in eval_indices]

        return inputs_train, inputs_eval, outputs_train, outputs_eval

        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text = self.combined[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        # Squeeze the batch dimension for each sample
        return {key: value.squeeze(0) for key, value in encoding.items()}



    def get_item_untokenized(self,idx):
        return self.combined[idx]

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)


        cls_embedding = outputs.last_hidden_state[:, 0, :] 

        return cls_embedding.squeeze(0) 
    
    def get_inputs_outputs(self):
        inputs_file = os.path.join(self.data_dir, 'input.txt')
        outputs_file = os.path.join(self.data_dir, 'output.txt')
        

        with open(inputs_file, 'r', encoding='utf-8') as f:
            inputs = f.readlines()
        
        with open(outputs_file, 'r', encoding='utf-8') as f:
            outputs = f.readlines()
        inputs = [line.strip() for line in inputs]
        outputs = [line.strip() for line in outputs]
        combined = [f"{input_text}, {output_text}" for input_text, output_text in zip(inputs, outputs)]
        return inputs, outputs, combined
    
        #returns the divided data, for evalaution and training
    def get_eval_train(self):
        os.makedirs('train_dir', exist_ok=True)
        os.makedirs('eval_dir', exist_ok=True)

        # Save train data to text files
        self.save_to_file(os.path.join('train_dir', "input.txt"), self.inputs_train)
        self.save_to_file(os.path.join('train_dir', "output.txt"), self.outputs_train)

        # Save evaluation data to text files
        self.save_to_file(os.path.join('eval_dir', "input.txt"), self.inputs_eval)
        self.save_to_file(os.path.join('eval_dir', "output.txt"), self.outputs_eval)

        return 'train_dir', 'eval_dir'
    
    #Helper function to write a list to a file, one sentence per line.
    def save_to_file(self, file_path, data_list):
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence in data_list:
                f.write(sentence.strip() + "\n")

