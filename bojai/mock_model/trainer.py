from abc import ABC, abstractmethod

#decides which trainer to use, depending on the task. Used by the train stage
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
        if self.task_type == '@TODO enter-model-name':
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
        if task_type == '@TODO enter-model-name':
            ''''
            @TODO 
            return a list of the names of hyper-parameters needed for your model's training. Leave empty if no 
            hyper-parameters are needed. 

            Example: 
            ['num_workers', 'learning_rate', 'batch_size', 'num_epochs'] 
            
            '''
            return []
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def edit_hyper_params(self, new_hyperparams : dict):
        self.hyperparams = new_hyperparams
        self.start_model()

            

#abstract class that is used as ab ase for other trainers. It dynamically assigns the hyper-params 
# Each Model must come with a required hyper-params set added to the manager above.         
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
    def train(self, qthread, progress_worker, loss-worker):
        pass

    @abstractmethod
    def evaluate(self, eval_dataset = None):
        pass

'''
@TODO 

Add your trainer, it must extend Trainer and implement train and evaluate. 
Each Model must come with a required hyper-params set added to the manager above.   

train is the training loop and evalaute is a way to evaluate model output. 

progrss_worker is a pyqtSignal, use to report training progress. Use to signal if needed. 
Example: 
progress = int((current_step / total_steps) * 100)
progrss_worker.emit(progress)  # Emit signal to UI

loss_worker is a pyqtSignal, use to report loss. 
Example:
avg_loss = total_loss / len(self.train_loader)
loss_updated.emit(avg_loss)

Once you signal workers, you need to allow UI to refresh using qthread. Copy-paste this line: 
qthread.msleep(1) 
'''
class ImplementYourTrainer(Trainer): 
    pass

