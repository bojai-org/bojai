'''
This import must stay here, do not remove it. 
'''
from processor import Processor

# ────────────────────────────────────────────────────────────────────────────────────────────────────
# YourDataProcessor: You will need to implement this class. Do not change its name or what it extends. 
# ────────────────────────────────────────────────────────────────────────────────────────────────────
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
    

'''
Finishing this class will allow you to build and use your first stage of the pipeline "prepare". 
'''