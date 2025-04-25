import os
import builtins
import pytest
from initialiseCLI import (
    initialize_pipeline_cli,
    browseDict,
    options
)

@pytest.fixture(autouse=True)
def reset_browse_dict():
    browseDict["options"] = 0
    browseDict["options-where"] = 0
    yield

@pytest.fixture
def fake_inputs(monkeypatch):
    def simulate_input(inputs):
        iterator = iter(inputs)
        monkeypatch.setattr(builtins, "input", lambda _: next(iterator))
    return simulate_input

@pytest.fixture
def patch_core(monkeypatch, tmp_path):
    monkeypatch.setattr("initialiseCLI.getNewModel", lambda: "FAKE_MODEL")
    monkeypatch.setattr("initialiseCLI.getNewTokenizer", lambda: "FAKE_TOKENIZER")
    monkeypatch.setattr("initialiseCLI.prepare_cli", lambda prep: print("📦 PREP CLI CALLED"))
    return tmp_path

def test_successful_run(fake_inputs, patch_core, capsys):
    fake_inputs([
        "TestModel",               # model name
        str(patch_core),          # data path
        "0.8",                     # train split
        "0.2",                     # eval split
        "x"                        # skip data prep
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out

    assert "✅ Pipeline successfully initialized!" in out
    assert "Model Name       : TestModel" in out
    assert "Train/Eval Split : 0.8/0.2" in out
    assert "Exiting Bojai CLI" in out

def test_triggers_data_prep(fake_inputs, patch_core, monkeypatch, capsys):
    monkeypatch.setattr("initialiseCLI.prepare_cli", lambda prep: print("📦 DATA PREP TRIGGERED"))
    fake_inputs([
        "GoModel",
        str(patch_core),
        "0.6",
        "0.4",
        "p"
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out
    assert "📦 DATA PREP TRIGGERED" in out

def test_invalid_path(fake_inputs, monkeypatch, capsys):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    fake_inputs([
        "BadModel",
        "/bad/path"
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out
    assert "❌ Path does not exist. Exiting." in out

def test_invalid_split(fake_inputs, patch_core, capsys):
    fake_inputs([
        "SplitModel",
        str(patch_core),
        "0.3",
        "0.3"
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out
    assert "❌ Train and eval splits must add to 1.0. Exiting." in out

def test_architecture_selection_success(fake_inputs, patch_core, monkeypatch, capsys):
    # Enable architecture selection
    monkeypatch.setitem(browseDict, "options", 1)
    monkeypatch.setitem(options, "mock_arch", lambda: "TokenizerInstance")
    
    fake_inputs([
        "ArchModel",
        str(patch_core),
        "0.5",
        "0.5",
        "1",  # Choose architecture index 1 (mock_arch)
        "x"
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out
    assert "Tokenizer        : bert" in out

def test_architecture_selection_invalid_index(fake_inputs, patch_core, monkeypatch, capsys):
    monkeypatch.setitem(browseDict, "options", 1)
    monkeypatch.setitem(options, "mock_arch", lambda: "TokenizerInstance")

    fake_inputs([
        "InvalidArchModel",
        str(patch_core),
        "0.5",
        "0.5",
        "99"  # Invalid index
    ])

    initialize_pipeline_cli()
    out = capsys.readouterr().out
    assert "❌ Invalid architecture number. Exiting." in out
