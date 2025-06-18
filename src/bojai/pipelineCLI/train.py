from prepare import Prepare
import psutil
import torch
import requests
import importlib.util
from cryptography.fernet import Fernet
from trainer import TrainingManager
from global_vars import init_model


# 2nd stage of training an ML model, used for training stage to train and initially evaluate the model, hyper-params should match the format in TrainingManager.
class Train:
    def __init__(self, prep: Prepare, hyper_params):
        self.hyper_params = hyper_params
        self.prep = prep
        self.trained = False
        if self.prep.prep_ready:
            self.eval, self.training, self.model, self.tokenizer, self.device = (
                prep.get_training_tools()
            )
        else:
            raise ValueError("preparation stage did not finish, cannot start training")
        init_model(self.prep.data, self.model, hyper_params)
        self.trainerManager = TrainingManager(
            prep.task_type,
            self.model,
            self.eval,
            self.training,
            self.device,
            self.tokenizer,
            hyper_params,
        )
        self.trainerManager.initialise()
        self.task_type = self.prep.task_type

    # updates the values of hyper_parameters
    def edit_hyperparams(self, new_hyperparams):
        self.trainerManager.edit_hyper_params(new_hyperparams)
        self.hyper_params = new_hyperparams

    # trains the model
    def train(self):
        self.trainerManager.trainer.train()
        self.trained = True

    # returns the evaluation error
    def get_eval_error(self):
        return self.trainerManager.trainer.evaluate()

    # returns how much of the device's memory is used, used during training.
    def get_device_use(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (
                1024**3
            )  # Convert to GB
            total = torch.cuda.get_device_properties(self.device).total_memory / (
                1024**3
            )  # Convert to GB
            return f"GPU: {allocated:.2f} GB / {total:.2f} GB used"

        elif self.device.type == "cpu":
            process = psutil.Process()
            ram_used = process.memory_info().rss / (
                1024**3
            )  # Resident Set Size (RSS) in GB
            total_ram = psutil.virtual_memory().total / (
                1024**3
            )  # Total system RAM in GB
            return f"CPU: {ram_used:.2f} GB / {total_ram:.2f} GB used"

        return "Unknown device type"
