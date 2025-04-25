import builtins
import pytest
from deployCLI import BojaiDeployCLI

class DummyTrainer:
    def __init__(self):
        self.model = type("FakeModel", (), {
            "state_dict": lambda self: {"weights": [1, 2, 3]}
        })()
        self.prep = "prep_obj"

class DummyDeploy:
    def __init__(self):
        self.trainer = DummyTrainer()
        self.max_length = 0
        self.new_data = None
        self.eval_data_path = None

    def update_eval_data(self, path):
        self.eval_data_path = path
        self.new_data = "mock_data"

    def get_eval_score(self, which):
        return 0.95 if which == 0 else 0.85

    def use_model(self, text_input):
        return f"ModelOutput({text_input})"

@pytest.fixture
def cli_instance():
    return BojaiDeployCLI(
        deploy=DummyDeploy(),
        train_cli_fn=lambda train: print("✅ Training CLI"),
        prepare_cli_fn=lambda prep: print("✅ Prepare CLI")
    )

@pytest.fixture
def simulate_input(monkeypatch):
    def _simulate(inputs):
        it = iter(inputs)
        monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return _simulate


def test_add_eval_data(cli_instance, simulate_input, capsys):
    simulate_input(["a", "new/path", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ New evaluation data added." in out


def test_evaluate_original(cli_instance, simulate_input, capsys):
    simulate_input(["e", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Accuracy: 0.95" in out


def test_evaluate_new(cli_instance, simulate_input, capsys):
    # add new data first
    cli_instance.deploy.new_data = True
    simulate_input(["n", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Accuracy: 0.85" in out


def test_use_model(cli_instance, simulate_input, capsys):
    simulate_input(["u", "Hello, model!", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Output: ModelOutput(Hello, model!)" in out


def test_save_model(monkeypatch, cli_instance, simulate_input, capsys):
    monkeypatch.setattr("deployCLI.torch", type("torch", (), {
        "save": lambda obj, path: print(f"[torch.save to {path}]")
    }))
    simulate_input(["s", "/models", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "[torch.save to /models/FakeModel.bin]" in out
    assert "✅ Model saved to /models/FakeModel.bin" in out


def test_return_to_training(cli_instance, simulate_input, capsys):
    simulate_input(["t", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Training CLI" in out
    assert "✅ Returned to deployment stage." in out


def test_return_to_preparation(cli_instance, simulate_input, capsys):
    simulate_input(["p", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Prepare CLI" in out
    assert "✅ Returned to deployment stage." in out


def test_invalid_choice(cli_instance, simulate_input, capsys):
    simulate_input(["x", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Invalid choice." in out


def test_quit(cli_instance, simulate_input, capsys):
    simulate_input(["q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "👋 Exiting deployment CLI." in out