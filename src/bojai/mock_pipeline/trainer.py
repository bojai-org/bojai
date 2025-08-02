from abc import ABC, abstractmethod
# ───────────────────────────────────────────────────────────────────────────────
# TrainingManager: Chooses the right Trainer for your model and passes it what it needs
# ───────────────────────────────────────────────────────────────────────────────
"""
📦 The TrainingManager is the brain of the training stage.
It decides which Trainer to use based on your model's task type, and ensures it has:
    - The right model
    - The right data
    - The right hyperparameters
    - The right setup (device, tokenizer, etc.)

Every model you create must have a corresponding entry in this manager.

When you call `initialise()`, the manager:
1. Looks up what hyperparameters are required for your model.
2. Validates them.
3. Instantiates your Trainer class with all the pieces.

🧠 This allows Bojai to dynamically support ANY type of trainer — you can plug in whatever training logic you want.

🔁 If you change the hyperparameters mid-session, call `edit_hyper_params()` to reinitialize the trainer with the new values.
"""


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
        self.logger = Logger()

    def initialise(self):
        self.start_model()
        self.initiated = True

    def start_model(self):
        required_keys = self.get_required_hyperparams(self.task_type)

        # Validate and extract required hyperparameters
        missing_keys = [key for key in required_keys if key not in self.hyperparams]
        if missing_keys:
            raise ValueError(f"Missing required hyperparameters: {missing_keys}")

        # Populate seq2seqhyper_params with validated hyperparameters
        hyper_params = {key: self.hyperparams[key] for key in required_keys}

        # Now pass the validated hyperparameters to TrainerSeq2Seq
        from custom_trainer import ImplementYourTrainer
        self.trainer = ImplementYourTrainer(
            self.model,
            self.training,
            self.eval,
            self.device,
            self.tokenizer,
            hyper_params,
        )

    def get_required_hyperparams(self, task_type):
        # Define required hyperparameters based on task_type
        return list(self.hyperparams.keys())

    def edit_hyper_params(self, new_hyperparams: dict):
        self.hyperparams = new_hyperparams
        self.start_model()


class Trainer(ABC):
    def __init__(self, model, device, tokenizer, hyper_params: dict):
        super().__init__()
        # Dynamically assign each hyperparameter as an instance attribute
        for key, value in hyper_params.items():
            setattr(self, key, value)
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.hyper_params = hyper_params  # Store the dict, not dict_keys
        self.logger = Logger()

    @abstractmethod
    def train(self, qthread, progress_worker, loss_worker):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

    @abstractmethod
    def evaluate(self, eval_dataset=None):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

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