"""
This file acts as the control panel for the BojAI pipeline.

It defines:
- Model and tokenizer loading logic
- Task-specific hyperparameters
- Configuration flags for UI behavior
- Optional external validation APIs
"""

# 🧠 Task Identifier
task_type = "gegt"

# 🔧 Training Configuration
hyper_params = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "num_workers": 0,
}

# 🎛️ Controls for UI and CLI behavior
browseDict = {
    "train": False,  # Does the training tab need a file browser?
    "prep": False,  # Does the prepare tab need a browser?
    "deploy_new_data": False,  # Does the deploy tab take new data from a browser?
    "use_model_upload": True,  # Does the model take a file input from user?
    "use_model_text": "Enter one picture to see output",  # Label in UI
    "init": False,  # Whether to call init_model before training
    "type": 0,  # 0=image, 1=voice, 2=text/numbers
    "eval_matrice": "perplexity",  # Evaluation metric to show in UI
    "options": 0,  # Whether the user needs to select from options
    "options-where": -1,  # Where options are applied: -1=nowhere, 0=tokenizer, 1=model
}


# 🧠 Load Tokenizers
def getNewTokenizer():
    """
    Returns the required tokenizers or processors.
    Can be a list if multiple modalities are used.
    """
    pass


# 🧠 Load Model
def getNewModel():
    """
    Returns a new instance of the model to be used in training or deployment.
    You can access tokenizers via getNewTokenizer() if needed for dimensions.
    """
    pass


# 🛠 Model Initialization Hook
# Only used if your model needs access to the dataset or hyperparameters to be fully initialized.
# Otherwise, skip this function and leave it as pass.
def init_model(data, model, hyper_params):
    """
    Call this function to finalize model initialization that depends on data or hyperparameters.

    This is necessary if your model requires:
    - Input/output dimensions that depend on the dataset
    - Hyperparameter-controlled architecture changes
    - Layer freezing or reweighting based on user config

    getNewModel() should return a base model object, but if it requires final setup using
    the dataset or hyperparameters, implement an `.init(data, hyper_params)` method inside your model class
    and call it here.

    Example:
        def init_model(data, model, hyper_params):
            model.init(data, hyper_params)

    If your model is fully defined without needing any external input, you can leave this function as `pass`.
    """
    pass


# class used for demonstration purposes only
class ObjectOfOption:
    pass


# options for tokenization or model selection. If you want to use this make sure to turn it on in browseDict dictionary.

options = {"name of option": ObjectOfOption}
