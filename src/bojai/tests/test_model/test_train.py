import pytest
import torch
from train import Train  
from prepare import Prepare


class DummyProcessor:
    def __init__(self):
        self.data = ["sample1", "sample2", "sample3"]

    def get_item_untokenized(self, idx):
        return self.data[idx]

    def __getitem__(self, idx):
        return f"TOK_{self.data[idx]}"

    def applyOp(self, op_name):
        self.data = [f"{op_name}({d})" for d in self.data]

    def __len__(self):
        return len(self.data)

class DummyData:
    def __init__(self):
        self.processor = DummyProcessor()
        self.train = "TRAIN_DATA"
        self.eval = "EVAL_DATA"

class DummyFormatter:
    def __init__(self):
        self.data_dir = "formatted/data/path"

    def __call__(self, task_type, data_dir, data_sep):
        self.data_dir = f"{data_dir}/checked"

class DummyTrainer:
    def __init__(self):
        self.trained = False

    def train(self):
        self.trained = True

    def evaluate(self):
        return 0.42

class DummyManager:
    def __init__(self):
        self.trainer = DummyTrainer()
        self.params = {}

    def initialise(self):
        pass

    def edit_hyper_params(self, params):
        self.params = params

@pytest.fixture
def prepare_instance():
    return Prepare(
        model_name="number",
        model="FakeModel",
        device=torch.device("cpu"),
        tokenizer="FakeTokenizer",
        data_dir="original/data",
        task_type="classification",
        division=[0.8, 0.2],
        data_sep=",",
        mock_processor=DummyData(),
        mock_formatter=DummyFormatter()
    )

@pytest.fixture
def train_instance(prepare_instance):
    return Train(prep=prepare_instance, hyper_params={"lr": 0.01}, mock_manager=DummyManager())

def test_train_initialization(train_instance):
    assert train_instance.eval == "EVAL_DATA"
    assert train_instance.training == "TRAIN_DATA"
    assert train_instance.model == "FakeModel"
    assert train_instance.tokenizer == "FakeTokenizer"
    assert train_instance.device.type == "cpu"

def test_edit_hyperparams(train_instance):
    new_params = {"lr": 0.001, "batch_size": 16}
    train_instance.edit_hyperparams(new_params)
    assert train_instance.hyper_params == new_params
    assert train_instance.trainerManager.params == new_params

def test_train_executes_training(train_instance):
    train_instance.train()
    assert train_instance.trained
    assert train_instance.trainerManager.trainer.trained

def test_get_eval_error(train_instance):
    error = train_instance.get_eval_error()
    assert error == 0.42

def test_get_device_use_cpu(train_instance):
    usage = train_instance.get_device_use()
    assert usage.startswith("CPU:")
