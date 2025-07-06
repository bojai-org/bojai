import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from processor import Processor



class YourDataProcessor(Processor):
    def __init__(
        self,
        data_dir,
        division,
        model,
        device,
        tokenizer,
        is_main=True,
        inputs=None,
        outputs=None,
    ):
        super().__init__(
            data_dir, division, model, device, tokenizer, is_main, inputs, outputs
        )

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

    def get_train_eval(self):

        num_samples = len(self.outputs)
        train_size = int(num_samples * self.division[0])

        # Randomly shuffle the indices
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        # Split inputs (n*d matrix) and outputs (n*1 vector)
        inputs_train = self.inputs[train_indices]
        inputs_eval = self.inputs[eval_indices]
        outputs_train = self.outputs[train_indices]  # Outputs for training
        outputs_eval = self.outputs[eval_indices]  # Outputs for evaluation

        return inputs_train, inputs_eval, outputs_train, outputs_eval

    def __getitem__(self, idx):
        input = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return input, output

    def __len__(self):
        return len(self.outputs)

    def get_item_untokenized(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]

        return input, output
