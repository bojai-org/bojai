from abc import ABC, abstractmethod
import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
import numpy as np
from processor import Processor
import requests

#decides which trainer to use, depending on the task. Used by the train stage
class TrainingManager:
    def __init__(self, task_type, model, eval, training, device, tokenizer, hyper_params: dict):
        self.trainer: Trainer = None
        self.initiated = False
        self.task_type = task_type
        self.model = model 
        self.eval = eval
        self.training = training
        self.device = device
        self.tokenizer = tokenizer
        self.hyperparams = hyper_params



    def initialise(self):
        self.start_model()
        self.initiated = True
    
    def start_model(self):
        required_keys = self.get_required_hyperparams(self.task_type)
        if self.task_type == 'get':
            # Validate and extract required hyperparameters
            missing_keys = [key for key in required_keys if key not in self.hyperparams]
            if missing_keys:
                raise ValueError(f"Missing required hyperparameters: {missing_keys}")

            # Populate seq2seqhyper_params with validated hyperparameters
            seq2seqhyper_params = {key: self.hyperparams[key] for key in required_keys}

            # Now pass the validated hyperparameters to TrainerSeq2Seq
            self.trainer = TrainerGET(self.model, self.training, self.eval, self.device, self.tokenizer, seq2seqhyper_params)

    def get_required_hyperparams(self, task_type):
        # Define required hyperparameters based on task_type
        if task_type == 'get':
            return ['num_workers', 'learning_rate', 'batch_size', 'num_epochs']
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def edit_hyper_params(self, new_hyperparams : dict):
        self.hyperparams = new_hyperparams
        self.start_model()

            

#abstract class that is used as ab ase for other trainers. It dynamically assigns the hyper-params 
# Each Model must come with a required hyper-params set added to the manager above.         
class Trainer(ABC):
    def __init__(self, model, device, tokenizer, hyper_params : dict):
        super().__init__()
        # Dynamically assign each hyperparameter as an instance attribute
        for key, value in hyper_params.items():
            setattr(self, key, value)
        self.model = model 
        self.device = device
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params.keys()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, eval_dataset = None):
        pass

#trainer for fine-tuning, used in GET medium and large. uses cross-entropy loss and Adam optimizer. Uses perplexity as eval metrice. 
class TrainerGET(Trainer): 
    def __init__(self, model, train_dataset, eval_dataset, device, tokenizer, hyperparams):
        super().__init__(model, device, tokenizer, hyperparams)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        padding_idx = tokenizer.pad_token_id  # Get padding token index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    
    def train(self, qthread = None , progress_updated = None, loss_updated = None):        
        total_steps = self.num_epochs  # Total steps in training
        current_step = 0  # Track progress

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                # Move input to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Shift logits and labels for sequence prediction
                logits = outputs
                shift_logits = logits[:, :-1, :].contiguous()  # Remove last token from logits
                shift_labels = input_ids[:, 1:].contiguous()  # Remove first token from input_ids

                # Compute loss
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()

                # Backward pass and optimizer step
                loss.backward()
                self.optimizer.step()

            # Update progress bar after each batch
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_updated.emit(progress)  # Emit signal to UI
            qthread.msleep(1)  # Allow UI to refresh

            # Print average loss for the epoch
            avg_loss = total_loss / len(self.train_loader)
            loss_updated.emit(avg_loss)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    
    def evaluate(self, eval_dataset=None):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_tokens = 1
        if eval_dataset is None:
            val_loader = self.val_loader
        else:
            val_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs  # Assuming first element is logits

                # Shift logits and labels for sequence prediction
                shift_logits = logits[:, :-1, :].contiguous()  # Remove last token from logits
                shift_labels = input_ids[:, 1:].contiguous()  # Remove first token from input_ids

                # Calculate loss (CrossEntropyLoss)
                padding_idx = self.tokenizer.pad_token_id  # Get the padding token index from the tokenizer

                # CrossEntropyLoss with ignore_index
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)  # Ignore padding tokens
                )

                total_loss += loss.item()
                total_tokens += shift_labels.numel()  # Total number of tokens

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()

