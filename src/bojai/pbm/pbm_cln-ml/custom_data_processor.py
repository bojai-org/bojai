import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from processor import Processor



class YourDataProcessor(Processor, Dataset):
    def __init__(self, data_dir, division, model, device, tokenizer, main=True):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.input_matrix, self.output = self.get_inputs_outputs(self.data_dir)
        train, eval = division
        if train + eval != 1:
            raise ValueError("the division must add to 1")

        if main:
            (
                self.inputs_train,
                self.inputs_eval,
                self.outputs_train,
                self.outputs_eval,
            ) = self.divide()
            self.train_dir, self.eval_dir = self.get_eval_train()

    def get_inputs_outputs(self, data_dir):
        with open(self.data_dir, "r") as file:
            matrix = json.load(file)

        # Assuming the matrix is stored as a list of lists in the JSON

        if matrix is None:
            raise ValueError("The JSON file does not contain a valid matrix.")

        matrix = np.array(matrix)

        if matrix.size != 0:
            inputs = matrix[:, :-1]
            output = matrix[:, -1]

        return inputs, output

    def divide(self):

        num_samples = len(self.output)
        train_size = int(num_samples * self.division[0])

        # Randomly shuffle the indices
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        # Split inputs (n*d matrix) and outputs (n*1 vector)
        inputs_train = self.input_matrix[train_indices]
        inputs_eval = self.input_matrix[eval_indices]
        outputs_train = self.output[train_indices]  # Outputs for training
        outputs_eval = self.output[eval_indices]  # Outputs for evaluation

        return inputs_train, inputs_eval, outputs_train, outputs_eval

    def __getitem__(self, idx):
        input = torch.tensor(self.input_matrix[idx], dtype=torch.float32)
        output = torch.tensor(self.output[idx], dtype=torch.float32)
        return input, output

    def __len__(self):
        return len(self.output)

    def get_item_untokenized(self, idx):
        input = self.input_matrix[idx]
        output = self.output[idx]

        return input, output

        # returns the divided data, for evalaution and training

    def get_eval_train(self):
        os.makedirs("train_dir", exist_ok=True)
        os.makedirs("eval_dir", exist_ok=True)

        # Save train data to text files
        self.save_to_file(
            os.path.join("train_dir", "data.json"),
            self.inputs_train,
            self.outputs_train,
        )

        # Save evaluation data to text files
        self.save_to_file(
            os.path.join("eval_dir", "data.json"), self.inputs_eval, self.outputs_eval
        )

        return os.path.join("train_dir", "data.json"), os.path.join(
            "eval_dir", "data.json"
        )

    # Helper function to write a matrix to a file.
    def save_to_file(self, file_path, data_list, output_list):
        output_list = output_list.reshape(-1, 1)
        combined_matrix = np.concatenate((data_list, output_list), axis=1)
        matrix_list = combined_matrix.tolist()

        # Save to JSON
        with open(file_path, "w") as f:
            json.dump(matrix_list, f)
