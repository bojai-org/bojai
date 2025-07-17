from processor import Processor
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


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
        self.image_processor = self.tokenizer[0]
        self.tokenizer_text = self.tokenizer[1]

    def get_inputs_outputs(self, data_dir):
        images = []
        image_dir = os.path.join(self.data_dir, "input")
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
            ):
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path)
                images.append(img)
            else:
                raise ValueError("All data in input directory must be images")

        outputs_file = os.path.join(self.data_dir, "output.txt")
        with open(outputs_file, "r", encoding="utf-8") as f:
            outputs = f.readlines()
        outputs = [line.strip() for line in outputs]

        return images, outputs

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

    def __getitem__(self, idx):
        image_features = self.image_processor(
            self.inputs[idx], return_tensors="pt"
        ).pixel_values
        labels = self.tokenizer_text(
            self.outputs[idx],
            return_tensors="pt",
            max_length=46,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        ).input_ids
        labels = labels.squeeze(0)

        return image_features.squeeze(0), labels

    def __len__(self):
        return len(self.outputs)
    
    def get_item_untokenized(self, idx):
        return self.inputs[idx], self.outputs[idx]