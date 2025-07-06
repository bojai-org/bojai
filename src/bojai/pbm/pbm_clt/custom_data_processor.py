
import torch
import os
from torch.utils.data import Dataset
import numpy as np
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


    def get_train_eval(self):
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