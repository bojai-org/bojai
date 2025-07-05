from abc import ABC, abstractmethod


# manages which data processor to use, used by the prep stage
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
        from custom_data_processor import YourDataProcessor
        if task_type == "gegt":
            self.processor = YourDataProcessor(
                self.data_dir, self.division, self.model, self.device, self.tokenizer
            )
            self.train = YourDataProcessor(
                self.processor.train_dir,
                [0, 1],
                self.model,
                self.device,
                self.tokenizer,
                False,
            )
            self.eval = YourDataProcessor(
                self.processor.eval_dir,
                [1, 0],
                self.model,
                self.device,
                self.tokenizer,
                False,
            )


# abstract class serving as the base for the other types of processors.
# The classes extending it should:
# - should split data
# - be able to accept some ops on data if they process numbers
# - should tokenize data
# - should shuffle data
# - should handle missing data and duplicates
class Processor(ABC):
    def __init__(self, data_dir, division, model, device, tokenizer):
        super().__init__()

        self.data_dir = data_dir
        self.division = division
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.inputs_train, self.inputs_eval, self.outputs_train, self.outputs_eval = (
            None,
            None,
            None,
            None,
        )

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_item_untokenized(self, idx):
        pass



