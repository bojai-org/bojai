import builtins
import pytest
from prepareCLI import BojaiPreparationCLI


class DummyPrep:
    def __init__(self):
        self.model_name = "DummyModel"
        self.model = object()
        self.tokenizer = object()
        self.num_data_points = 3
        self.prep_ready = True

    def check_tokenized(self, idx=None):
        return f"TOK_{idx if idx is not None else 'RANDOM'}"

    def view_raw_data(self, idx=None):
        return f"RAW_{idx if idx is not None else 'RANDOM'}"

    def update_data(self, new_path):
        self.num_data_points = 5


@pytest.fixture
def cli_instance():
    prep = DummyPrep()
    return BojaiPreparationCLI(
        prep=prep,
        train_cls=lambda prep, hparams: "FAKE_TRAINER",
        train_cli_fn=lambda trainer: print("🚀 TRAIN CLI STARTED"),
        browse_dict={"type": 2},
        hyperparams={"lr": 0.01}
    )


@pytest.fixture
def simulate_input(monkeypatch):
    def _simulate(inputs):
        iterator = iter(inputs)
        monkeypatch.setattr(builtins, "input", lambda _: next(iterator))
    return _simulate


def test_view_tokenized_and_quit(cli_instance, simulate_input, capsys):
    simulate_input(["v", "", "q", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "TOK_RANDOM" in out
    assert "Exiting data preparation stage" in out


def test_view_raw_and_quit(cli_instance, simulate_input, capsys):
    simulate_input(["r", "", "q", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "RAW_RANDOM" in out


def test_update_data_path(cli_instance, simulate_input, capsys):
    simulate_input(["u", "new/path", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Data updated successfully. Total data points: 5" in out


def test_empty_data_path_rejected(cli_instance, simulate_input, capsys):
    simulate_input(["u", "", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Path cannot be empty." in out


def test_trigger_training(cli_instance, simulate_input, capsys):
    simulate_input(["t", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "🚀 TRAIN CLI STARTED" in out
    assert "✅ Prep complete" in out


def test_invalid_choice(cli_instance, simulate_input, capsys):
    simulate_input(["x", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Invalid option" in out
    assert "Exiting data preparation stage" in out


def test_training_blocked_if_not_ready(simulate_input, capsys):
    class NotReadyPrep(DummyPrep):
        def __init__(self):
            super().__init__()
            self.prep_ready = False

    cli = BojaiPreparationCLI(
        prep=NotReadyPrep(),
        train_cli_fn=lambda trainer: print("should not appear"),
    )
    simulate_input(["t", "q"])
    cli.run()
    out = capsys.readouterr().out
    assert "❌ Prep is not ready. Cannot continue to training." in out
