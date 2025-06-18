'''
This import must stay here, do not remove it. 
'''
from trainer import Trainer

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ImplementYourTrainer: Insert your training and evaluation logic here. Do not change its name or what it extends. 
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
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

class ImplementYourTrainer(Trainer):
    def __init__(
        self, model, training_data, eval_data, device, tokenizer, hyper_params: dict
    ):
        super().__init__(model, device, tokenizer, hyper_params)

        # 🧠 Store your data and unpack hyperparameters here
        self.training_data = training_data
        self.eval_data = eval_data

        # Example: self.batch_size = self.batch_size
        # You can initialize optimizers, metrics, or caches here if needed

    def train(self, qthread, progress_worker, loss_worker):
        """
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
        """

        raise NotImplementedError("Implement your training loop")

    def evaluate(self, eval_dataset=None):
        """
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
        """

        raise NotImplementedError("Implement your evaluation loop")
    

'''
Finishing this class will allow you to build and use your second stage of the pipeline "train". 
'''