from train import Train
import sys

#3rd stage of training an ML model, deploys the model after training or the model as-is
class Deploy():
    #max_legnth: the maximum length of the output. 
    def __init__(self, trainer : Train, max_length = None, mock_user = None, mock_processor = None):
        self.trainer = trainer
        self.prep = self.trainer.prep
        self.max_length = max_length
        self.manager = mock_user
        self.mock_processor = mock_processor
        self.new_data = None

    #evaluates the data. If 0, uses the inital evaluation, if 1 uses uploaded evaluation dataset. 
    def get_eval_score(self, which_one):
        if which_one == 0: 
            return self.trainer.get_eval_error()
        if which_one == 1: 
            return self.trainer.trainerManager.trainer.evaluate(self.new_data.processor)
    
    #adds a new eval dataset. 
    def update_eval_data(self, new_data_dir):

        revised_dir = self.prep.check_data_match(new_data_dir, self.prep.task_type, self.prep.data_sep)
        new_data = self.mock_processor(revised_dir)
        self.new_data = new_data
    
    #gets the model output
    def use_model(self, input):
        sys.stdout.reconfigure(encoding='utf-8')
        return self.manager.user.use_model(input)



        
