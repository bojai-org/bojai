"""
User classes for handling different pipeline types during prediction.
"""

import torch
from PIL import Image
import io
import numpy as np
from .logging_utils import get_logger
logger = get_logger(__name__)

class User:
    """Base class for model users"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        if hasattr(self.model, 'to'):
            self.model.to(device)

class UserCLI(User):
    """CLI pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction (image input)"""
        # Accepts file path or bytes
        if isinstance(input, str):
            image = Image.open(input).convert("RGB")
        elif isinstance(input, bytes):
            image = Image.open(io.BytesIO(input)).convert("RGB")
        else:
            raise ValueError("Input must be a file path or image bytes")
        image = image.resize((500, 500))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
        tensor = torch.tensor(arr).unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        return {"prediction": pred, "confidence": confidence}

class UserCLN(User):
    """CLN pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction (numerical input)"""
        # Accepts list or np.ndarray
        x = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)
            pred = int(torch.round(output).item())
            confidence = float(torch.sigmoid(output).item())
        return {"prediction": pred, "confidence": confidence}

class UserCLNML(User):
    """CLN-ML pipeline user implementation"""
    def __init__(self, model, tokenizer=None, device="cpu", max_length=None):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input):
        """Use the model for prediction (advanced ML input)"""
        if not callable(self.model):
            raise RuntimeError("Model is not callable - invalid model file")
        
        # Accepts list or np.ndarray
        x = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)
            pred = int(torch.round(output).item())
            confidence = float(torch.sigmoid(output).item())
            metadata = {"processing_time": 0.0, "feature_importance": []}
        return {"prediction": pred, "confidence": confidence, "metadata": metadata} 