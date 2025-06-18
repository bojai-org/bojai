"""
User classes for handling different pipeline types during prediction.
"""

import torch
from PIL import Image

class User:
    """Base class for model users"""
    def __init__(self, model, tokenizer, device, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

class UserCLI(User):
    """CLI pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction"""
        # 1. Handle input (file path or image data)
        # 2. Convert to RGB
        # 3. Resize to (500, 500)
        # 4. Convert to tensor and normalize
        # 5. Add batch dimension
        # 6. Run model inference
        # 7. Get prediction class
        # 8. Return prediction
        pass

class UserCLN(User):
    """CLN pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction"""
        # 1. Run model inference
        # 2. Round output
        # 3. Return prediction
        pass

class UserCLNML(User):
    """CLN-ML pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction"""
        # 1. Run model inference
        # 2. Round output
        # 3. Return prediction
        pass 