import pytest
from prepare import Prepare  # Adjust the import path if needed

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

@pytest.fixture
def prepare_instance():
    return Prepare(
        model_name="number",
        model="FakeModel",
        device="cpu",
        tokenizer="FakeTokenizer",
        data_dir="original/data",
        task_type="classification",
        division=[0.8, 0.2],
        data_sep=",",
        mock_processor=DummyData(),
        mock_formatter=DummyFormatter()
    )

def test_initialization(prepare_instance):
    assert prepare_instance.prep_ready
    assert prepare_instance.data_dir == "original/data/checked"
    assert prepare_instance.model == "FakeModel"

def test_view_raw_data_with_index(prepare_instance):
    result = prepare_instance.view_raw_data(index=1)
    assert result == "sample2"

def test_view_raw_data_random_index(prepare_instance):
    result = prepare_instance.view_raw_data()
    assert result in ["sample1", "sample2", "sample3"]

def test_check_tokenized_with_index(prepare_instance):
    result = prepare_instance.check_tokenized(idx=0)
    assert result == "TOK_sample1"

def test_check_tokenized_random_index(prepare_instance):
    result = prepare_instance.check_tokenized()
    assert result.startswith("TOK_sample")

def test_apply_op_on_data_only_for_number_model(prepare_instance):
    prepare_instance.apply_op_on_data("normalize")
    assert all(["normalize(" in d for d in prepare_instance.data.processor.data])

def test_apply_op_skipped_for_non_number():
    instance = Prepare(
        model_name="text",
        model="FakeModel",
        device="cpu",
        tokenizer="FakeTokenizer",
        data_dir="original/data",
        task_type="classification",
        division=[0.8, 0.2],
        data_sep=",",
        mock_processor=DummyData(),
        mock_formatter=DummyFormatter()
    )
    before = list(instance.data.processor.data)
    instance.apply_op_on_data("normalize")
    assert instance.data.processor.data == before  # unchanged

def test_get_training_tools_returns_all(prepare_instance):
    tools = prepare_instance.get_training_tools()
    assert tools == ("EVAL_DATA", "TRAIN_DATA", "FakeModel", "FakeTokenizer", "cpu")

def test_update_data_replaces_pipeline(prepare_instance):
    new_processor = DummyData()
    prepare_instance.update_data(data_dir="new/data", processor=new_processor)
    assert prepare_instance.data_dir == "new/data/checked"
    assert prepare_instance.data == new_processor
    assert prepare_instance.prep_ready
