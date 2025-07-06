import os
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
            matrix = file.readlines()
        input = []
        output = []

        for item in matrix: 
            items = item.split(",")
            input.append(items[0])
            output.append(item[1])
        return input, output

    def get_train_eval(self):

        num_samples = len(self.outputs)
        train_size = int(num_samples * self.division[0])

        # Randomly shuffle the indices
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        # Split inputs (n*d matrix) and outputs (n*1 vector)
        inputs_train = self.inputs[train_indices]
        inputs_eval = self.inputs[eval_indices]
        outputs_train = self.outputs[train_indices]  # Outputs for training
        outputs_eval = self.outputs[eval_indices]  # Outputs for evaluation

        return inputs_train, inputs_eval, outputs_train, outputs_eval

    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]

        return input, output

    def __len__(self):
        return len(self.outputs)

    def get_item_untokenized(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]

        return input, output