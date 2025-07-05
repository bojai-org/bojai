
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from processor import Processor

class YourDataProcessor(Processor, Dataset):
    def __init__(
        self, data_dir, division, model, device, tokenizer, max_length, main=True
    ):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.inputs, self.outputs, self.combined = self.get_inputs_outputs(self.data_dir)
        train, eval = division
        if train + eval != 1:
            raise ValueError("the division must add to 1")
        self.max_length = max_length
        if main:
            (
                self.inputs_train,
                self.inputs_eval,
                self.outputs_train,
                self.outputs_eval,
            ) = self.divide()
            self.train_dir, self.eval_dir = self.get_eval_train()
        tokenizer.build_vocab(self.inputs)

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
        text = self.inputs[idx]
        label = self.outputs[idx]
        encoding = self.tokenizer.encode(text)
        return encoding, torch.tensor(label)

    def get_item_untokenized(self, idx):
        return self.combined[idx]

    def get_inputs_outputs(self, data_dir):
        inputs_file = os.path.join(self.data_dir, "input.txt")
        outputs_file = os.path.join(self.data_dir, "output.txt")

        with open(inputs_file, "r", encoding="utf-8") as f:
            inputs = f.readlines()

        with open(outputs_file, "r", encoding="utf-8") as f:
            outputs = f.readlines()

        inputs = [line.strip() for line in inputs]
        outputs = [int(line.strip()) for line in outputs]
        for i in range(len(outputs)):
            if outputs[i] == 2:
                outputs[i] = 0
        combined = [
            f"input : {input_text}, output : {output_text}"
            for input_text, output_text in zip(inputs, outputs)
        ]
        return inputs, outputs, combined

        # returns the divided data, for evalaution and training

    def get_eval_train(self):
        os.makedirs("train_dir", exist_ok=True)
        os.makedirs("eval_dir", exist_ok=True)

        # Save train data to text files
        self.save_to_file(os.path.join("train_dir", "input.txt"), self.inputs_train)
        self.save_to_file(os.path.join("train_dir", "output.txt"), self.outputs_train)

        # Save evaluation data to text files
        self.save_to_file(os.path.join("eval_dir", "input.txt"), self.inputs_eval)
        self.save_to_file(os.path.join("eval_dir", "output.txt"), self.outputs_eval)

        return "train_dir", "eval_dir"

    # Helper function to write a list to a file, one sentence per line.
    def save_to_file(self, file_path, data_list):
        with open(file_path, "w", encoding="utf-8") as f:
            for sentence in data_list:
                f.write(str(sentence).strip() + "\n")