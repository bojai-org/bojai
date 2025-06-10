from abc import ABC, abstractmethod
from torch.utils.data import Dataset

# ───────────────────────────────────────────────────────────────────────────────
# ProcessorManager: Chooses the right data processor and sets up train/eval sets
# ───────────────────────────────────────────────────────────────────────────────
"""
📦 The ProcessorManager is the entry point of the data preprocessing stage.
It handles:
    - Which data processor to use (based on your task type or setup)
    - How the dataset is split into training and evaluation sets
    - Passing the model, tokenizer, and device into the processor

🧠 You only need to create ONE processor class that inherits from `Processor`.
This class will be instantiated 3 times:
    1. As the full processor (loads and splits data)
    2. As the train dataset (wraps the train portion)
    3. As the eval dataset (wraps the eval portion)

🔁 Once you’ve implemented your processor, plug it into the `decide_which_processor` method below.
"""


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


# ───────────────────────────────────────────────────────────────────────────────
# YourDataProcessor: You will need to implement this class
# ───────────────────────────────────────────────────────────────────────────────
class YourDataProcessor(Processor):
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
        super().__init__(
            data_dir, division, model, device, tokenizer, is_main, inputs, outputs
        )

    def get_inputs_outputs(self, data_dir):
        """
        Load your data from the specified directory and return:
        - inputs: a list of input samples
        - outputs: a list of output labels (or targets)

        Example:
        return ["Translate this", "Another example"], ["Traduce esto", "Otro ejemplo"]
        """
        return [], []

    def get_train_eval(self):
        """
        Split self.inputs and self.outputs into four lists:
        - inputs_train
        - inputs_eval
        - outputs_train
        - outputs_eval

        Return them in the order listed above. You can use self.inputs and 
        self.outputs, which should be populated by get_inputs_outputs().

        Example:
        return self.inputs[:80], self.inputs[80:], self.outputs[:80], self.outputs[80:]
        """
        return [], [], [], []

    def __len__(self):
        """
        Return the number of examples in this dataset. 
        This default implementation should work correctly if 
        get_inputs_outputs() is implemented properly.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Return a tokenized version of the input/output at the given index.
        The format is up to you, as it will only be used in the model 
        you implement.

        Example:
        return {
            "input_ids": torch.tensor(...),
            "labels": torch.tensor(...)
        }
        """
        return None

    def get_item_untokenized(self, idx):
        """
        Return the raw (untokenized) input/output pair at the given index.
        The format is up to you; it will optionally be used by your model.
        It will also be used by the pipeline CLI and UI, where the return 
        value will be converted to a string.

        Example:
        return self.inputs[idx], self.outputs[idx]
        """
        return None