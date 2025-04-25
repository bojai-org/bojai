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
            return ["learning_rate", "num_epochs", "num_batches"]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def edit_hyper_params(self, new_hyperparams: dict):
        self.hyperparams = new_hyperparams
        if new_hyperparams["learning_rate"] != None:
            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = new_hyperparams["learning_rate"]
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

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.num_batches, shuffle=True
        )
        self.val_loader = DataLoader(
            eval_dataset, batch_size=self.num_batches, shuffle=False
        )
        self.loss_fn = nn.BCELoss()
        if model.__class__.__name__ != "kNN":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def train(self, qthread=None, progress_updated=None, loss_updated=None):
        self.num_batches = len(self.train_loader)  # Ensure correct batch count
        total_steps = len(self.train_loader)
        current_step = 0  # Track progress
        for epoch in range(len(self.train_loader)):
            total_loss = 0
            # forward pass and loss
            for x, y in self.train_loader:
                y_predicted, _ = self.model(x)
                y_predicted = y_predicted.to(torch.float32)
                y = y.to(torch.float32)
                y = y.unsqueeze(1)
                loss = self.loss_fn(y_predicted, y)
                total_loss += loss.item()

                # backward pass
                loss.backward()

                # updates
                self.optimizer.step()

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                progress_updated.emit(progress)  # Emit signal to UI
                qthread.msleep(1)  # Allow UI to refresh

            # Print average loss for the epoch
            avg_loss = total_loss / len(self.train_loader)
            loss_updated.emit(avg_loss)

    def evaluate(self, eval_dataset=None):
        self.model.eval()  # Set the model to evaluation mode
        correct_predictions = 0
        total_samples = 0

        # Use the provided eval dataset or fallback to the validation loader
        if eval_dataset is None:
            val_loader = self.val_loader
        else:
            val_loader = DataLoader(
                eval_dataset, batch_size=self.num_batches, shuffle=False
            )

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for x, y in val_loader:
                outputs, reg = self.model(x)
                y = y.unsqueeze(1).to(torch.float32)  # Ensure shape consistency

                predicted_labels = torch.round(
                    outputs
                )  # Convert probabilities to binary predictions
                correct_predictions += (
                    (predicted_labels == y).sum().item()
                )  # Count correct predictions
                total_samples += y.size(0)  # Add batch size to total samples

        # Return accuracy
        return correct_predictions / total_samples if total_samples > 0 else 0
