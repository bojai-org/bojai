from abc import ABC, abstractmethod
import torch
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer
from model import FineTunedTransformerGEGT

# ───────────────────────────────────────────────────────────────────────────────
# userManager: Selects which user (inference logic) to use, based on the task.
# ───────────────────────────────────────────────────────────────────────────────
'''
🎯 This class exists to pick the correct User class for your pipeline.

Just like how `TrainingManager` picks your Trainer, this picks your model User — the component responsible for using your trained model to:
- Accept a user input (text, image, etc.)
- Run it through the model
- Return a clean, readable output

This is called in the deploy stage of your pipeline.

🧠 You can define different "task types" and assign different user logic for each.

Once selected, the `self.user` attribute will be an instance of your custom `User` implementation, ready to use.
'''

class userManager():
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.tokenizer = tokenizer

        if task_type == '@TODO enter-model-name':
            # Replace the task_type above with your actual model name
            self.user = ImplementYourUser(model, tokenizer, device, max_length)


# ───────────────────────────────────────────────────────────────────────────────
# User: Abstract class for deploying your model
# ───────────────────────────────────────────────────────────────────────────────
'''
🧩 The `User` class defines how your model is *used* in deployment.

You must extend this class with your own implementation and override `use_model`.

This is NOT training logic — it’s how you process live input (from the UI or CLI),
pass it to the model, and return an output.

Use this for:
- Running inference
- Handling post-processing
- Supporting images, text, numbers, files — whatever your model uses

📦 You can load the model, apply preprocessing, and return formatted outputs here.
'''

class User(ABC):
    def __init__(self, model, tokenizer, device, max_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    @abstractmethod
    def use_model(self, input):
        '''
        input: string, image path, number, or custom data — whatever your model expects

        This method should:
        1. Preprocess the input
        2. Run it through the model (using self.model)
        3. Post-process and return the result

        Example (text task):
            tokens = self.tokenizer(input, return_tensors="pt")
            output = self.model(**tokens)
            return output

        Example (image task):
            image = Image.open(input)
            processed = self.processor(image).unsqueeze(0)
            output = self.model(processed)
            return postprocess(output)
        '''
        pass

# ───────────────────────────────────────────────────────────────────────────────
# ImplementYourUser: Your own model usage logic
# ───────────────────────────────────────────────────────────────────────────────
'''
✅ This is where YOU define how your model is used in deployment.

You must extend `User` and implement `use_model`.

You can use:
- Torch
- HuggingFace
- OpenCV
- Pure Python logic
- Anything that fits your model

This class is what the user interacts with during the deploy stage (via UI or CLI).
'''

class ImplementYourUser(User):
    def use_model(self, input):
        '''
        Replace this with your own usage logic.

        Example (for a text model):
            inputs = self.tokenizer(input, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            return outputs

        Example (for an image model):
            image = Image.open(input)
            processed = self.processor(image).unsqueeze(0)
            output = self.model(processed)
            return output

        ⚠️ This must return a value — either a string, number, or other result.
        '''
        pass
