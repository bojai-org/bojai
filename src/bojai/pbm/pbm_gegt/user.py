from abc import ABC, abstractmethod
import torch
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer
from model import FineTunedTransformerGEGT


# manages which data user to use depending on task. Used by the deploy stage.
class userManager:
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.tokenizer = tokenizer
        if task_type == "gegt":
            self.user = UserGet(model, tokenizer, device, max_length)


# abstract class for using the model and evaluating it, used as a base for specific data users
class User(ABC):
    def __init__(self, model, tokenizer, device, max_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    @abstractmethod
    def use_model(self, input):
        pass


# the dta user for the GET models.
class UserGet(User):
    def __init__(self, model, tokenizer, device, max_length):
        super().__init__(model, tokenizer, device, max_length)

    def use_model(self, input_image: Image.Image):
        # Preprocess the input image using ViTImageProcessor
        processor = self.tokenizer[0]
        image = Image.open(input_image)
        inputs = processor(images=image, return_tensors="pt")

        # Move tensors to the same device as the model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Set model to evaluation mode
        self.model.eval()

        # Create a tensor for the initial input_ids (for text generation, starting with <BOS>)
        start_token = self.tokenizer[
            1
        ].pad_token_id  # or another start token, depending on your setup
        input_ids = torch.tensor([[start_token]], device=self.device)

        # Forward pass through the model with pixel values and initial input_ids for the decoder
        with torch.no_grad():
            # Pass the image (pixel_values) and the input_ids (start token) into the model
            outputs = self.model(
                pixel_values=inputs["pixel_values"],
                input_ids=input_ids,  # image pixel values as input
            )

            # Get the logits from the model's decoder output
            logits = outputs

            # Get the predicted token ids (index with highest logit for each time step)
            predicted_token_ids = torch.argmax(logits, dim=-1)

            # Decode the generated token ids into text
            generated_text = self.tokenizer[1].decode(
                predicted_token_ids[0], skip_special_tokens=True
            )

        return generated_text


