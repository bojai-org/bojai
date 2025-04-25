# test_deploy.py
import pytest
from deploy import Deploy
from train import Train
from prepare import Prepare
import torch

# ---------------------- MOCK CLASSES ----------------------

class DummyProcessor:
    def __init__(self, prefix="sample"):
        self.data = [f"{prefix}{i}" for i in range(3)]

    def get_item_untokenized(self, idx):
        return self.data[idx]

    def __getitem__(self, idx):
        return f"TOK_{self.data[idx]}"

    def applyOp(self, op_name):
        self.data = [f"{op_name}({d})" for d in self.data]

    def __len__(self):
        return len(self.data)

class DummyData:
    def __init__(self, prefix="sample"):
        self.processor = DummyProcessor(prefix=prefix)
        self.train = "TRAIN_DATA"
        self.eval = "EVAL_DATA"

class DummyFormatter:
    def __init__(self):
        self.data_dir = ""

    def __call__(self, task_type, data_dir, data_sep):
        self.data_dir = f"{data_dir}/checked"

class DummyTrainer:
    def train(self):
        pass

    def evaluate(self, data=None):
        if data:
            return 0.99
        return 0.42

class DummyManager:
    def __init__(self):
        self.trainer = DummyTrainer()

    def initialise(self):
        pass

    def edit_hyper_params(self, params):
        pass

class DummyUser:
    class InnerUser:
        def use_model(self, input):
            return f"Output for {input}"

    def __init__(self):
        self.user = self.InnerUser()

# ---------------------- FIXTURES ----------------------

@pytest.fixture
def prepare_instance():
    return Prepare(
        model_name="text",
        model="FakeModel",
        device=torch.device("cpu"),
        tokenizer="FakeTokenizer",
        data_dir="data",
        task_type="classification",
        division=[0.8, 0.2],
        data_sep=",",
        mock_processor=DummyData(),
        mock_formatter=DummyFormatter()
    )

@pytest.fixture
def train_instance(prepare_instance):
    return Train(
        prep=prepare_instance,
        hyper_params={"lr": 0.01},
        mock_manager=DummyManager()
    )

@pytest.fixture
def deploy_instance(train_instance):
    return Deploy(
        trainer=train_instance,
        max_length=50,
        mock_user=DummyUser(),
        mock_processor=lambda path: DummyData(prefix="eval")
    )

# ---------------------- TESTS ----------------------

def test_deploy_initialization(deploy_instance):
    assert deploy_instance.max_length == 50
    assert deploy_instance.prep is deploy_instance.trainer.prep

def test_get_eval_score_from_initial(deploy_instance):
    score = deploy_instance.get_eval_score(which_one=0)
    assert score == 0.42

def test_update_eval_data_and_get_eval_score(deploy_instance):
    deploy_instance.update_eval_data("new/eval")
    assert deploy_instance.new_data is not None
    assert all(x.startswith("eval") for x in deploy_instance.new_data.processor.data)
    score = deploy_instance.get_eval_score(which_one=1)
    assert score == 0.99

def test_use_model(deploy_instance):
    output = deploy_instance.use_model("hello")
    assert output == "Output for hello"
