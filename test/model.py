"""
This is your custom model file.

✅ You are free to use ANY architecture here:
- PyTorch model (e.g., CNN, Transformer, MLP)
- Sklearn model (e.g., LogisticRegression, SVM)
- Even hand-coded rules or NumPy-based logic

There are NO base classes, NO requirements, and NO restrictions.
Just make sure the object you define here can be passed to your trainer class.

👇 Example of a flexible placeholder model:
"""


class YourModel:
    def __init__(self):
        # Initialize your layers, sklearn estimator, or logic here
        self.model = None  # Replace with actual implementation

    def predict(self, x):
        """
        not required
        Run forward inference on a single input or batch
        This method is optional — only implement if your trainer uses it
        """
        pass

    def save(self, path):
        """
        not required
        Save your model to disk (optional).
        You can use torch.save, joblib, pickle, etc.
        """
        pass

    def load(self, path):
        """
        not required
        Load your model from disk (optional).
        """
        pass


"""
You’ll use this model object in:
- train.py for training and evaluation
- deploy.py for inference
You can modify it however you want.
"""
