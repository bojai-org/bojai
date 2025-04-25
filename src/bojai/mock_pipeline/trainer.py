from abc import ABC, abstractmethod

# ───────────────────────────────────────────────────────────────────────────────
# TrainingManager: Chooses the right Trainer for your model and passes it what it needs
# ───────────────────────────────────────────────────────────────────────────────
'''
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
'''
class TrainingManager:
    def __init__(self, task_type, model, eval, training, device, tokenizer, hyper_params: dict):
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
        self.trainer = ImplementYourTrainer(self.model, self.training, self.eval, self.device, self.tokenizer, hyper_params)

    def get_required_hyperparams(self, task_type):
        # Define required hyperparameters based on task_type
        return []
    
    def edit_hyper_params(self, new_hyperparams : dict):
        self.hyperparams = new_hyperparams
        self.start_model()

            

# ───────────────────────────────────────────────────────────────────────────────
# Trainer: Abstract base class for all trainers
# ───────────────────────────────────────────────────────────────────────────────
'''
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
''' 
class Trainer(ABC):
    def __init__(self, model, device, tokenizer, hyper_params : dict):
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
        pass

    @abstractmethod
    def evaluate(self, eval_dataset = None):
        pass


class ImplementYourTrainer(Trainer):
    def __init__(self, model, training_data, eval_data, device, tokenizer, hyper_params: dict):
        super().__init__(model, device, tokenizer, hyper_params)

        # 🧠 Store your data and unpack hyperparameters here
        self.training_data = training_data
        self.eval_data = eval_data

        # Example: self.batch_size = self.batch_size
        # You can initialize optimizers, metrics, or caches here if needed

    def train(self, qthread, progress_worker, loss_worker):
        '''
        This is your training loop. You are free to implement it however you like.
        
        🧩 Tips:
        - Loop through self.training_data however fits your model
        - Use self.model to call your forward or fit logic
        - Use self.device and self.tokenizer if needed
        - Emit progress with:   progress_worker.emit(percent)
        - Emit loss with:       loss_worker.emit(loss_value)
        - Call qthread.msleep(1) to allow UI to refresh
        
        Example (pseudo-code):
        total_steps = len(self.training_data)
        for i, data in enumerate(self.training_data):
            loss = self.model.train_step(data)
            progress = int((i + 1) / total_steps * 100)
            progress_worker.emit(progress)
            loss_worker.emit(loss)
            qthread.msleep(1)
        '''

        raise NotImplementedError("Implement your training loop")

    def evaluate(self, eval_dataset=None):
        '''
        This is your evaluation function. You can calculate accuracy, BLEU, MSE — anything.

        If eval_dataset is provided, use that instead of self.eval_data.
        
        Example (pseudo-code):
        total = 0
        correct = 0
        for data in eval_dataset:
            prediction = self.model.predict(data)
            correct += (prediction == data.label)
            total += 1
        return correct / total
        '''

        raise NotImplementedError("Implement your evaluation loop")
