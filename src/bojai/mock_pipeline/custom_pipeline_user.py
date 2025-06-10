'''
This import must stay here, do not remove it. 
'''
from user import User
# ────────────────────────────────────────────────────────────────────────────────────────
# ImplementYourUser: Your own model usage logic. Do not change its name or what it extends. 
# ────────────────────────────────────────────────────────────────────────────────────────
"""
✅ This is where YOU define how your model is used in deployment.

You must extend `User` and implement `use_model`.

You can use:
- Torch
- HuggingFace
- OpenCV
- Pure Python logic
- Call non-python files
- Anything that fits your model

This class is what the user interacts with during the deploy stage (via UI or CLI).
"""


class ImplementYourUser(User):
    def use_model(self, input):
        """
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
        """
        raise NotImplementedError("Implement your usage logic")
    

'''
Finishing this class will allow you to build and use your third stage of the pipeline "deploy". 
'''