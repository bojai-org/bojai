from abc import ABC, abstractmethod
from torch.utils.data import Dataset
# ───────────────────────────────────────────────────────────────────────────────
# ProcessorManager: Chooses the right data processor and sets up train/eval sets
# ───────────────────────────────────────────────────────────────────────────────


class ProcessorManager:
    def __init__(self, data_dir, division, model, device, tokenizer, task_type):
        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.processor = None
        self.decide_which_processor(task_type)

    def decide_which_processor(self, task_type):
        # Replace this with logic to choose between multiple processors if needed
        from custom_data_processor import YourDataProcessor
        self.processor = YourDataProcessor(
            self.data_dir, self.division, self.model, self.device, self.tokenizer
        )
        self.train = YourDataProcessor(
            None,
            [0, 1],
            self.model,
            self.device,
            self.tokenizer,
            is_main=False,
            inputs=self.processor.inputs_train,
            outputs=self.processor.outputs_train,
        )
        self.eval = YourDataProcessor(
            None,
            [1, 0],
            self.model,
            self.device,
            self.tokenizer,
            is_main=False,
            inputs=self.processor.inputs_eval,
            outputs=self.processor.outputs_eval,
        )


# ───────────────────────────────────────────────────────────────────────────────
# Processor: Abstract base class for all dataset processors
# ───────────────────────────────────────────────────────────────────────────────
"""
🧠 Every custom data processor must inherit from this class.

It defines the interface Bojai expects for handling data, and automatically handles:
- Storing inputs/outputs
- Train/eval division
- Accessing tokenized and untokenized samples

You MUST implement the following functions in the non-abstract class:
- get_inputs_outputs(): Load your data and return two lists: inputs, outputs
- get_train_eval(): Split those lists into train/eval portions
- __getitem__(): Return a tokenized example
- get_item_untokenized(): Return a raw untokenized example
- __len__(): Return the number of examples

📌 NOTE:
- Inputs/outputs can be any type (text, image paths, numbers…).
- Tokenization logic is up to you.
"""


class Processor(ABC, Dataset):
    def __init__(
        self,
        data_dir,
        division,
        model,
        device,
        tokenizer,
        is_main=True,
        inputs=None,
        outputs=None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        if is_main:
            self.inputs, self.outputs = self.get_inputs_outputs(data_dir)
            (
                self.inputs_train,
                self.inputs_eval,
                self.outputs_train,
                self.outputs_eval,
            ) = self.get_train_eval()
        else:
            self.inputs = inputs
            self.outputs = outputs

    @abstractmethod
    def get_inputs_outputs(self, data_dir):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

    @abstractmethod
    def get_train_eval(self):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

    @abstractmethod
    def __getitem__(self, idx):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass

    @abstractmethod
    def get_item_untokenized(self, idx):
        '''
        Abstract definition, do not touch. Go to the non-abstract class below. 
        '''
        pass


