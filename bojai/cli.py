import argparse
import os
import shutil
import subprocess
from pathlib import Path


SHARED_FILES = [
    "bojai/model/deploy.py",
    "bojai/model/deployUI.py",
    "bojai/model/initialiseUI.py",
    "bojai/model/prepare.json",
    "bojai/model/prepareUI.py",
    "bojai/model/train.py", 
    "bojai/model/trainUI.py"
]


def init_model_workspace(model_name):
    workspace_dir = f"applets/{model_name}"
    if os.path.exists(workspace_dir):
        print(f"Workspace '{model_name}' already exists.")
        return

    os.makedirs(workspace_dir)
    print(f"{workspace_dir} directory created, now copying files")

    # Copy shared files
    for file_path in os.listdir("model"):
        filename = file_path
        file_path = os.path.join("model", filename)
        shutil.copy(file_path, f"{workspace_dir}/{filename}")
        print(f"copied {filename}")

    # Copy model-specific files
    if os.path.exists(model_name):
        for file in os.listdir(model_name):
            filename = file
            file_path = os.path.join(model_name, file)
            shutil.copy(file_path, f"{workspace_dir}/{filename}")
            print(f"copied {filename}")
    else:
        print(f"Warning: Model-specific directory '{model_name}' not found.")
        return
    
    print(f"Initialized workspace for model '{model_name}' at ./{model_name}/")


def launch_model(model_name):
    workspace_dir = Path(f"applets/{model_name}")
    initialise_file = workspace_dir / "initialiseUI.py"

    if not initialise_file.exists():
        raise FileNotFoundError(f"Cannot find 'initialiseUI.py' in {workspace_dir}/")

    subprocess.run(["python", "initialiseUI.py"], cwd=workspace_dir)



def train_model(model_name):
    print(f"[Training] Starting training loop for model: {model_name}")
    # Placeholder logic for training


def evaluate_model(model_name):
    print(f"[Evaluation] Evaluating model: {model_name}")
    # Placeholder logic for evaluation


def main():
    parser = argparse.ArgumentParser(description="BojAi Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bojai start
    parser_start = subparsers.add_parser("start", help="Start a model")
    parser_start.add_argument("--model", required=True, help="Model to start (e.g., get, summarizer)")

    # bojai train
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument("--model", required=True, help="Model to train")

    # bojai evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a model")
    parser_eval.add_argument("--model", required=True, help="Model to evaluate")

    # bojai init
    parser_init = subparsers.add_parser("init", help="Initialize a model workspace")
    parser_init.add_argument("--model", required=True, help="Model to initialize (e.g., get, summarizer)")

    args = parser.parse_args()

    if args.command == "start":
        launch_model(args.model)
    elif args.command == "train":
        train_model(args.model)
    elif args.command == "evaluate":
        evaluate_model(args.model)
    elif args.command == "init":
        init_model_workspace(args.model)


if __name__ == "__main__":
    main()
