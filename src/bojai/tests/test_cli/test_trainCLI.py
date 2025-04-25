import builtins
import pytest
from trainCLI import BojaiTrainingCLI
import sys


class DummyTrainer:
    def __init__(self):
        self.trained = False
        self.model = type("MockModel", (), {})()

    def train(self, qthread, progress_cb, loss_cb):
        print("🔥 Mock training...")
        self.trained = True

    def evaluate(self):
        return 0.95


class DummyTrain:
    def __init__(self):
        self.model = type("DummyModel", (), {})()
        self.prep = type("Prep", (), {"device": "cpu", "data": "some_data"})()
        self.trainerManager = type(
            "Manager",
            (),
            {"hyperparams": {"lr": 0.01, "epochs": 5}, "trainer": DummyTrainer()},
        )()

    def edit_hyperparams(self, new_params):
        self.trainerManager.hyperparams = new_params


@pytest.fixture
def simulate_input(monkeypatch):
    def _simulate(inputs):
        it = iter(inputs)
        monkeypatch.setattr(builtins, "input", lambda _: next(it))

    return _simulate


@pytest.fixture
def cli_instance():
    return BojaiTrainingCLI(train=DummyTrain())


def test_training(cli_instance, simulate_input, capsys):
    simulate_input(["t", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "🔥 Mock training..." in out
    assert "✅ Training complete." in out


def test_update_hyperparams(cli_instance, simulate_input, capsys):
    simulate_input(["u", "0.05", "10", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Hyperparameters updated." in out
    assert cli_instance.train.trainerManager.hyperparams == {"lr": 0.05, "epochs": 10}


def test_invalid_hyperparam_input(cli_instance, simulate_input, capsys):
    simulate_input(["u", "abc", "", "q"])  # bad input for lr
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Invalid value for lr" in out
    assert "✅ Hyperparameters updated." in out


def test_evaluation(cli_instance, simulate_input, capsys):
    simulate_input(["e", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Evaluation Result: Accuracy = 0.95" in out


def test_replace_model_confirmed(cli_instance, simulate_input, capsys):
    simulate_input(["r", "y", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "✅ Model successfully reset." in out


def test_replace_model_cancelled(cli_instance, simulate_input, capsys):
    simulate_input(["r", "n", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Cancelled." in out


def test_deploy_blocked_before_training(cli_instance, simulate_input, capsys):
    simulate_input(["d", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Please train your model first before deploying." in out


def test_deploy_after_training(cli_instance, simulate_input, monkeypatch, capsys):
    class DummyDeploy:
        def __init__(self, train, max_length):
            print("🚀 Deploy created")

    monkeypatch.setattr("trainCLI.Deploy", DummyDeploy)
    monkeypatch.setattr(
        "trainCLI.deploy_cli", lambda deploy: print("📦 [deploy_cli called]")
    )

    simulate_input(["t", "d", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "📦 [deploy_cli called]" in out


def test_invalid_choice(cli_instance, simulate_input, capsys):
    simulate_input(["x", "q"])
    cli_instance.run()
    out = capsys.readouterr().out
    assert "❌ Invalid choice." in out
    assert "👋 Exiting training stage." in out


def test_return_to_prepare(monkeypatch, simulate_input, capsys):
    simulate_input(["p", "q"])

    # Mock module and class to patch the import in trainCLI.run
    class DummyPrepRunner:
        def run(self):
            print("🔁 Mocked prepareCLI returned")

    dummy_prepare_module = type(
        "prepareCLI", (), {"BojaiPreparationCLI": lambda prep: DummyPrepRunner()}
    )
    monkeypatch.setitem(sys.modules, "prepareCLI", dummy_prepare_module)

    cli = BojaiTrainingCLI(train=DummyTrain())
    cli.run()
    out = capsys.readouterr().out
    assert "🔁 Mocked prepareCLI returned" in out
