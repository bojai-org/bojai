from abc import ABC, abstractmethod
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from processor import Processor
class YourDataProcessor(Processor, Dataset):
    def __init__(self, data_dir, division, model, device, tokenizer, main=True):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.images, self.labels = self.get_inputs_outputs(self.data_dir)
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
        outputs = [int(line.strip()) for line in outputs]

        return images, outputs

    def divide(self):
        # Determine the indices for splitting
        num_samples = len(self.labels)
        train_size = int(num_samples * self.division[0])  # Training set size
        indices = np.random.permutation(num_samples)  # Randomly shuffle the indices
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        # Split inputs and outputs
        inputs_train = [self.images[i] for i in train_indices]
        inputs_eval = [self.images[i] for i in eval_indices]
        outputs_train = [self.labels[i] for i in train_indices]
        outputs_eval = [self.labels[i] for i in eval_indices]

        return inputs_train, inputs_eval, outputs_train, outputs_eval

    def __getitem__(self, idx):
        img = self.images[idx]
        img = img.resize((500, 500))
        img_array = np.array(img)
        # Convert the NumPy array to a PyTorch tensor
        # The shape of the image_array will be (H, W, 3), so we need to transpose it to (3, H, W)
        img_tensor = (
            torch.tensor(img_array).permute(2, 0, 1).float()
        )  # Convert to float for neural network processing

        # Normalize the tensor (optional)
        img_tensor = img_tensor / 255.0  # Scaling the pixel values to the range [0, 1]

        return img_tensor, self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def get_item_untokenized(self, idx):
        return self.images[idx], self.labels[idx]

        # returns the divided data, for evalaution and training

    def get_eval_train(self):
        os.makedirs("train_dir", exist_ok=True)
        os.makedirs("eval_dir", exist_ok=True)

        # Save train data to text files
        self.save_to_file_image(os.path.join("train_dir", "input"), self.inputs_train)
        self.save_to_file(os.path.join("train_dir", "output.txt"), self.outputs_train)

        # Save evaluation data to text files
        self.save_to_file_image(os.path.join("eval_dir", "input"), self.inputs_eval)
        self.save_to_file(os.path.join("eval_dir", "output.txt"), self.outputs_eval)

        return "train_dir", "eval_dir"

    # Helper function to write a list to a file, one sentence per line.
    def save_to_file(self, file_path, data_list):
        with open(file_path, "w", encoding="utf-8") as f:
            for sentence in data_list:
                f.write(str(sentence) + "\n")

    def save_to_file_image(self, file_path, data_list):
        if not os.path.exists(file_path):
            os.makedirs(file_path)  # Create directory if it doesn't exist

        for i, img in enumerate(data_list, start=1):
            dir_path = os.path.join(file_path, f"{i}.jpg")  # Save as .jpg
            img.save(dir_path, format="JPEG")  # You can change format to PNG if needed
