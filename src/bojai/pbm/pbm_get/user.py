from abc import ABC, abstractmethod
import torch


# manages which data user to use depending on task. Used by the deploy stage.
class userManager:
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.device = device
        self.tokenizer = tokenizer
        if task_type == "get":
            self.user = UserGet(model, tokenizer, max_length)


# abstract class for using the model and evaluating it, used as a base for specific data users
class User(ABC):
    def __init__(self, model, tokenizer, max_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @abstractmethod
    def use_model(self, input):
        pass


# the dta user for the GET models.
class UserGet(User):
    def __init__(self, model, tokenizer, max_length):
        super().__init__(model, tokenizer, max_length)

    def use_model(self, input_text):
        # Tokenize input
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Check if self.max_length is None and assign a default value if needed
        if self.max_length is None:
            self.max_length = 50  # Set a default value for max_length

        # Check if input_ids has the expected shape
        if len(input_ids.shape) < 2:
            raise ValueError("input_ids must have at least 2 dimensions")

        # Set model to evaluation mode
        self.model.eval()

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Start generating tokens from the input
        generated_ids = input_ids.clone()

        for i in range(self.max_length - input_ids.shape[1]):  # Use 'i' instead of '_'
            with torch.no_grad():
                logits = self.model(
                    input_ids=generated_ids, attention_mask=attention_mask
                )

            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)

            predicted_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)

            if predicted_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, predicted_token_id], dim=1)

            new_mask = torch.ones_like(predicted_token_id, device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        return generated_text
