from train import Train
from processor import ProcessorManager
from user import userManager

import sys


# 3rd stage of training an ML model, deploys the model after training or the model as-is
class Deploy:
    # max_legnth: the maximum length of the output.
    def __init__(self, trainer: Train, max_length=None):
        self.trainer = trainer
        self.prep = self.trainer.prep
        self.max_length = max_length
        self.manager = userManager(
            self.trainer.task_type,
            self.trainer.model,
            self.trainer.tokenizer,
            self.trainer.device,
            max_length,
        )
        self.new_data = None

    # evaluates the data. If 0, uses the inital evaluation, if 1 uses uploaded evaluation dataset.
    def get_eval_score(self, which_one):
        if which_one == 0:
            return self.trainer.get_eval_error()
        if which_one == 1:
            return self.trainer.trainerManager.trainer.evaluate(self.new_data.processor)

    # adds a new eval dataset.
    def update_eval_data(self, new_data_dir):

        
        new_data: ProcessorManager = ProcessorManager(
            new_data_dir,
            [0, 1],
            self.trainer.model,
            self.trainer.device,
            self.trainer.tokenizer,
            self.trainer.task_type,
        )
        self.new_data = new_data

    # gets the model output
    def use_model(self, input):
        sys.stdout.reconfigure(encoding="utf-8")
        return self.manager.user.use_model(input)
