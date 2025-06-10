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
        return []

    def edit_hyper_params(self, new_hyperparams: dict):
        self.hyperparams = new_hyperparams
        self.start_model()


# ───────────────────────────────────────────────────────────────────────────────
# Trainer: Abstract base class for all trainers
# ───────────────────────────────────────────────────────────────────────────────
"""
🧠 Every custom Trainer must extend this abstract class.

It ensures consistency across all trainers, and automatically sets up:
- The model
- The device
- The tokenizer
- Any hyperparameters you defined (they are auto-assigned as instance variables)

You MUST implement:
- train()
- evaluate()

This lets Bojai treat all trainers the same way, even though the inside can be 100% custom.
"""


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



