from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# decides which trainer to use, depending on the task. Used by the train stage
class TrainingManager:
    def __init__(
        self, task_type, model, eval, training, device, tokenizer, hyper_params: dict
    ):
        self.trainer: Trainer = None
        self.initiated = False
        self.task_type = task_type
        self.model = model
        self.eval = eval
        self.training = training
        self.device = device
        self.tokenizer = tokenizer
        self.hyperparams = hyper_params

    def initialise(self):
        self.start_model()
        self.initiated = True

    def start_model(self):
        required_keys = self.get_required_hyperparams(self.task_type)
        if self.task_type == "cln":
            # Validate and extract required hyperparameters
            missing_keys = [key for key in required_keys if key not in self.hyperparams]
            if missing_keys:
                raise ValueError(f"Missing required hyperparameters: {missing_keys}")

            # Populate seq2seqhyper_params with validated hyperparameters
            seq2seqhyper_params = {key: self.hyperparams[key] for key in required_keys}

            # Now pass the validated hyperparameters to TrainerSeq2Seq
            self.trainer = TrainerCLN(
                self.model,
                self.training,
                self.eval,
                self.device,
                self.tokenizer,
                seq2seqhyper_params,
            )

    def get_required_hyperparams(self, task_type):
        # Define required hyperparameters based on task_type
        if task_type == "cln":
            return ["learning_rate"]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def edit_hyper_params(self, new_hyperparams: dict):
        self.hyperparams = new_hyperparams
        self.start_model()


# abstract class that is used as ab ase for other trainers. It dynamically assigns the hyper-params
# Each Model must come with a required hyper-params set added to the manager above.
class Trainer(ABC):
    def __init__(self, model, device, tokenizer, hyper_params: dict):
        super().__init__()
        # Dynamically assign each hyperparameter as an instance attribute
        for key, value in hyper_params.items():
            setattr(self, key, value)
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params.keys()
        self.logger = Logger()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, eval_dataset=None):
        pass


# trainer for fine-tuning, used in GET medium and large. uses cross-entropy loss and Adam optimizer. Uses perplexity as eval metrice.
class TrainerCLN(Trainer):
    def __init__(
        self, model, train_dataset, eval_dataset, device, tokenizer, hyperparams
    ):
        super().__init__(model, device, tokenizer, hyperparams)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    def train(self, qthread=None, progress_updated=None, loss_updated=None):
        self.num_batches = len(self.train_loader)  # Ensure correct batch count
        total_steps = len(self.train_loader)
        current_step = 0  # Track progress
        total_loss = 0
        # forward pass and loss
        for x, y in self.train_loader:
            y_predicted = self.model(x)
            y = y.unsqueeze(1)
            y = y.to(torch.float32)
            loss = self.loss_fn(y_predicted, y)
            total_loss += loss.item()

            # backward pass
            loss.backward()

            # updates
            self.optimizer.step()

            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_updated.emit(progress)  # Emit signal to UI
            valid = self.evaluate()
            qthread.msleep(1)  # Allow UI to refresh
            train_score = self.evaluate(self.train_dataset)
            self.logger.log(epoch=current_step-1, train=train_score, valid=valid, loss=loss.item())

        # Print average loss for the epoch
        avg_loss = total_loss / len(self.train_loader)
        loss_updated.emit(avg_loss)

    def evaluate(self, eval_dataset=None):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        num = 0
        # Use the provided eval dataset or fallback to the validation loader
        if eval_dataset is None:
            val_loader = self.val_loader
        else:
            
            val_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for x, y in val_loader:
                outputs = self.model(x)
                y = y.unsqueeze(1)
                y = y.to(torch.float32)
                if torch.equal(y, torch.round(outputs)):
                    total_loss += 1
                num += 1
        
        if eval_dataset != None:
            self.logger.log(eval_score= total_loss / num)
        # return loss
        return total_loss / num
''''
Singleton design pattern definition. This will be used as a tag to the Logger to turn it into a singleton class. 
'''
def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class Logger:
    """
    @ADD TO ALL trainer.py(s)
    Singleton logger to track per-epoch training/validation metrics, and final evaluation score.

    Logs:
        - self.logs[epoch] = {"train": ..., "valid": ...}
        - self.eval = final evaluation score (single scalar)
    """

    def __init__(self):
        self.logs = {}
        self.eval = None  # separate from epoch logging

    '''
    Logs either the final evaluation score or epoch-specific training/validation metrics.

    - If `eval_score` is provided, logs it as the final evaluation score.
    - If `epoch`, `train`, or `valid` are provided, logs them under the specified epoch.
    - If both `eval_score` and any of (`epoch`, `train`, `valid`) are given in the same call,
      raises an error to prevent optimization bias (evaluation must occur post-training).

    Behavior:
    - If the given epoch already exists, its values are updated with the new data.
    - If it doesn't exist, a new epoch entry is created with the provided values.
    '''
    def log(self, eval_score=None, epoch=None, train=None, valid=None, loss=None):
        if eval_score is not None and any(x is not None for x in [train, valid, epoch]):
            raise ValueError("Cannot log both eval_score and epoch-based logs in the same call. Read about Optimization bias in machine learning.")
        
        if eval_score is not None: 
            self.logs[-1] = eval_score
            return
        if epoch is None:
            raise ValueError("Epoch must be specified when logging train/valid metrics.")

        if epoch not in self.logs:
            self.logs[epoch] = {}

        if train is not None:
            self.logs[epoch]['train'] = train

        if valid is not None:
            self.logs[epoch]['valid'] = valid
        
        if loss is not None:
            self.logs[epoch]['loss'] = loss



    def log_eval(self, score: float):
        """Logs the one-time final evaluation score (not tied to epoch)."""
        self.eval = score

    def get_logs(self):
        return self.logs

    def get_eval(self):
        return self.eval

    def __str__(self):
        return f"Logs: {self.logs}\nEval: {self.eval}"

    def set_logger(self, logs):
        self.logs = logs