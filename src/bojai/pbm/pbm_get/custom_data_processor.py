from processor import Processor
import torch
import os
from torch.utils.data import Dataset
import numpy as np

# processes data for sequence to sequence models.
# the file must be stored in two txt files, one for input and one for output.
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
        text = self.combined[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Squeeze the batch dimension for each sample
        return {key: value.squeeze(0) for key, value in encoding.items()}

    def get_item_untokenized(self, idx):
        return self.combined[idx]

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding.squeeze(0)

    def get_inputs_outputs(self, data_dir):
        inputs_file = os.path.join(self.data_dir, "input.txt")
        outputs_file = os.path.join(self.data_dir, "output.txt")

        with open(inputs_file, "r", encoding="utf-8") as f:
            inputs = f.readlines()

        with open(outputs_file, "r", encoding="utf-8") as f:
            outputs = f.readlines()
        inputs = [line.strip() for line in inputs]
        outputs = [line.strip() for line in outputs]
        combined = [
            f"{input_text}, {output_text}"
            for input_text, output_text in zip(inputs, outputs)
        ]
        return inputs, outputs, combined