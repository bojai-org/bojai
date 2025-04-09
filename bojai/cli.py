import argparse
import os
import shutil
import subprocess
from pathlib import Path




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


def new_costum_model(model_name, dir):
    workspace_dir = os.path.join(dir, model_name)
    print(workspace_dir)
    if os.path.exists(workspace_dir):
        print(f"Workspace '{model_name}' already exists.")
        return

    os.makedirs(workspace_dir)
    print(f"{workspace_dir} directory created, now copying files")

    # Copy shared files
    for file_path in os.listdir("mock_model"):
        filename = file_path
        file_path = os.path.join("mock_model", filename)
        shutil.copy(file_path, f"{workspace_dir}/{filename}")
        print(f"copied {filename}")


def launch_model(model_name, stage):
    workspace_dir = Path(f"applets/{model_name}")

    if stage == 'all':
        initialise_file = workspace_dir / "initialiseUI.py"

        if not initialise_file.exists():
            raise FileNotFoundError(f"Cannot find 'initialiseUI.py' in {workspace_dir}/")

        subprocess.run(["python", "initialiseUI.py"], cwd=workspace_dir)
    
    elif stage == 'prepare':
        initialise_file = workspace_dir / "prepareUI.py"

        if not initialise_file.exists():
            raise FileNotFoundError(f"Cannot find 'prepareUI.py' in {workspace_dir}/")

        subprocess.run(["python", "prepareUI.py"], cwd=workspace_dir)

    elif stage == 'train':
        initialise_file = workspace_dir / "trainUI.py"

        if not initialise_file.exists():
            raise FileNotFoundError(f"Cannot find 'trainUI.py' in {workspace_dir}/")

        subprocess.run(["python", "trainUI.py"], cwd=workspace_dir)

    elif stage == 'deploy':
        initialise_file = workspace_dir / "deployUI.py"

        if not initialise_file.exists():
            raise FileNotFoundError(f"Cannot find 'deployUI.py' in {workspace_dir}/")

        subprocess.run(["python", "deployUI.py"], cwd=workspace_dir)
    
    else: 
        raise ValueError("stage name is wrong, must be one of prepare, train, or deploy.")



def remove_model(model_name):
    workspace_dir = Path(f"applets/{model_name}")
    if workspace_dir.exists() and workspace_dir.is_dir():
        try:
            shutil.rmtree(workspace_dir)
            print(f"Removed workspace for model '{model_name}' at ./{workspace_dir}/")
        except PermissionError as e:
            print(f"⚠️ Could not delete '{workspace_dir}': {e}")
            print("Make sure no files are open or being used by another process.")
    else:
        print(f"Workspace '{model_name}' does not exist.")



def main():
    parser = argparse.ArgumentParser(description="BojAi Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bojai start
    parser_start = subparsers.add_parser("start", help="Start a model")
    parser_start.add_argument("--model", required=True, help="Model to start (e.g., get, summarizer)")
    parser_start.add_argument("--stage", type=str, default='all', help="Port to run the model on (default: 8080)")

    # bojai init
    parser_init = subparsers.add_parser("build", help="Initialize a model workspace")
    parser_init.add_argument("--model", required=True, help="Model to initialize (e.g., get, summarizer)")

    # bojai evaluate
    parser_eval = subparsers.add_parser("remove", help="Remove a built model")
    parser_eval.add_argument("--model", required=True, help="Model to remove")

    # bojai evaluate
    parser_eval = subparsers.add_parser("create", help="Create a costum machine learning pipeline. Code your own processor, model, trainer, and deployer")
    parser_eval.add_argument("--model", required=True, help="give a name to your pipeline")
    parser_eval.add_argument("--directory", required=True, help="Enter where you want to access the code for your costum pipeline's directory")

    args = parser.parse_args()

    if args.command == "start":
        launch_model(args.model, args.stage)

    elif args.command == "build":
        init_model_workspace(args.model)
    
    elif args.command == "remove":
        remove_model(args.model)
    
    elif args.command == "create":
        new_costum_model(args.model, args.directory)

if __name__ == "__main__":
    main()
