from abc import ABC, abstractmethod

# ───────────────────────────────────────────────────────────────────────────────
# userManager: Selects which user (inference logic) to use, based on the task.
# ───────────────────────────────────────────────────────────────────────────────
"""
🎯 This class exists to pick the correct User class for your pipeline.

Just like how `TrainingManager` picks your Trainer, this picks your model User — the component responsible for using your trained model to:
- Accept a user input (text, image, etc.)
- Run it through the model
- Return a clean, readable output

This is called in the deploy stage of your pipeline.

🧠 You can define different "task types" and assign different user logic for each.

Once selected, the `self.user` attribute will be an instance of your custom `User` implementation, ready to use.
"""


class userManager:
    def __init__(self, task_type, model, tokenizer, device, max_length=None):
        self.user = None
        self.tokenizer = tokenizer

        # Replace the task_type above with your actual model name
        from custom_pipeline_user import ImplementYourUser
        self.user = ImplementYourUser(model, tokenizer, device, max_length)


# ───────────────────────────────────────────────────────────────────────────────
# User: Abstract class for deploying your model
# ───────────────────────────────────────────────────────────────────────────────
"""
🧩 The `User` class defines how your model is *used* in deployment.

You must extend this class with your own implementation and override `use_model`.

This is NOT training logic — it’s how you process live input (from the UI or CLI),
pass it to the model, and return an output.

Use this for:
- Running inference
- Handling post-processing
- Supporting images, text, numbers, files — whatever your model uses

📦 You can load the model, apply preprocessing, and return formatted outputs here.
"""


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
        Abstract method, do not touch. Go to the non-abstract class below to implement your logic. 
        '''
        pass
