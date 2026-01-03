from processor import ProcessorManager
import random
from global_vars import init_model
from agent import DataProcessorAgent
from timing_utils import Timer

# first stage of the machine learning process. Deals with data processing and manual handling
class Prepare:
    # data_type : medium, small, big ..etc
    # model: model to be used in trainig
    # device: the device used for training, ex GPU
    # tokenizer : the tokenizer that turns data into numbers
    # data_dir : directory in which data lives
    # task_type : the type of task, check codes in internal documentation
    # division: [a,b] where a+b = 1 and a is how much of the data for training, b how much for the data for evaluation
    # data_sep: data separator used if any.
    def __init__(
        self,
        model_name: str,
        model,
        device,
        tokenizer,
        data_dir: str,
        task_type: str,
        division,
        data_sep: str = None,
        use_agent = None
    ):
        with Timer("initialising Prepare phase"):
            self.prep_ready = False
            self.model = model
            self.device = device
            self.tokenizer = tokenizer
            self.data_dir = data_dir
            self.division = division
            self.task_type = task_type
            if use_agent[0]: 
                agent = DataProcessorAgent("custom_data_processor.py", self.data_dir)
                agent.process(use_agent[1], use_agent[2])

            self.data: ProcessorManager = ProcessorManager(
                self.data_dir, division, model, device, tokenizer, task_type
            )
            self.eval, self.train = self.data.eval, self.data.train
            self.num_data_points = len(self.data.processor)
            self.model_name = model_name
            self.task_type = task_type
            self.data_sep = data_sep
            self.prep_ready = True

    # returns a random datapoint
    def view_raw_data(self, index=None):
        if index == None:
            index = random.randint(0, self.num_data_points - 1)
        return self.data.processor.get_item_untokenized(index)

    # replaces the old data with new one, needs to repeat the matching process
    def update_data(self, data_dir, division=None):
        self.prep_ready = False
        data_dir = self.check_data_match(data_dir, self.task_type, self.data_sep)
        self.data_dir = data_dir
        div = division
        if div == None:
            div = self.division
        self.division = div
        self.data: ProcessorManager = ProcessorManager(
            self.data_dir, div, self.model, self.device, self.tokenizer, self.task_type
        )
        self.eval, self.train = self.data.eval, self.data.train
        self.num_data_points = len(self.data.processor)
        self.prep_ready = True

    # returns tokenized data
    def check_tokenized(self, idx=None):
        if idx == None:
            idx = random.randint(0, self.num_data_points - 1)
        return self.data.processor[idx]

    # applies a specific operation like scaling, normalizing. Only works if data is numbers
    def apply_op_on_data(self, op_name: str):
        self.prep_ready = False
        if self.model_name == "number":
            self.data.processor.applyOp(op_name)
        self.prep_ready = True

    # returns the data and the models, to be used in the training stage.
    def get_training_tools(self):
        return self.eval, self.train, self.model, self.tokenizer, self.device