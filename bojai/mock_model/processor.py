from abc import ABC, abstractmethod
from torch.utils.data import Dataset

#manages which data processor to use, used by the prep stage
class ProcessorManager():
    def __init__(self, data_dir, division, model, device, tokenizer, task_type):
        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.processor = None
        self.decide_which_processor(task_type)
    
    def decide_which_processor(self, task_type):
        if task_type == '@TODO model-name-here':
            self.processor = YourDataProcessor(self.data_dir, self.division, self.model, self.device, self.tokenizer)
            self.train =  YourDataProcessor(self.processor.train_dir, [0,1], self.model, self.device, self.tokenizer, False)
            self.eval =  YourDataProcessor(self.processor.eval_dir, [1,0], self.model, self.device, self.tokenizer, False)
    


#abstract class serving as the base for the other types of processors. 
class Processor(ABC, Dataset):
    def __init__(self, data_dir, division, model, device, tokenizer):
        super().__init__()
        
        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.inputs_train, self.inputs_eval, self.outputs_train, self.outputs_eval = None, None, None,None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_item_untokenized(self,idx):
        pass


class YourDataProcessor(Processor):
    def __init__(self, data_dir, division, model, device, tokenizer, is_main=True):
        super().__init__(data_dir, division, model, device, tokenizer)
        self.is_main = is_main
        self.inputs = []
        self.outputs = []

        # Load and split your data here
        # Example: CSV, JSON, TXT, folders, etc.
        # You are expected to assign:
        # - self.inputs_train
        # - self.inputs_eval
        # - self.outputs_train
        # - self.outputs_eval

        # self.inputs_train = ...
        # self.outputs_train = ...
        # self.inputs_eval = ...
        # self.outputs_eval = ...

    def __len__(self):
        # return total length depending on whether it's train or eval
        return len(self.inputs_train if self.is_main else self.inputs)

    def __getitem__(self, idx):
        # return tokenized input/output pair
        # example:
        # return {
        #     "input_ids": torch.tensor(...),
        #     "labels": torch.tensor(...)
        # }
        raise NotImplementedError("Implement __getitem__")

    def get_item_untokenized(self, idx):
        # return untokenized raw example, useful for debugging
        raise NotImplementedError("Implement get_item_untokenized")