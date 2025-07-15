import argparse
import shutil
import subprocess
import os
from pathlib import Path

# Base directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

def update_pipeline_logic(code: str) -> str:
    # Placeholder: Add custom logic here
    # e.g., inject new steps, modify config, etc.
    if "# New Step Placeholder" in code:
        return code.replace("# New Step Placeholder", "def new_step():\n    print('New step added')")
    
    # Implementation goes here
    pass
def build_pipeline(pipeline_name, replace):
    workspace_dir = SCRIPT_DIR / "applets" / f"bm_{pipeline_name}"

    if workspace_dir.exists() and not replace:
        print(f"Workspace '{pipeline_name}' already exists. Use --replace to rebuild.")
        return

    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"{workspace_dir} directory created, now copying files")
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    for folder in ["pipelineCLI", "pipelineUI"]:
        source_dir = SCRIPT_DIR / folder
        for file_path in source_dir.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, workspace_dir / file_path.name)
                print(f"copied {file_path.name}")

    specific_dir = SCRIPT_DIR / "pbm" / f"pbm_{pipeline_name}"
    if specific_dir.exists():
        for file_path in specific_dir.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, workspace_dir / file_path.name)
                print(f"copied {file_path.name}")
    else:
        print(f"Warning: pipeline-specific directory '{pipeline_name}' not found.")
        return

    print(f"Initialized workspace for pipeline '{pipeline_name}' at {workspace_dir}")


def build_dir(pipeline_name, directory, replace):
    workspace_dir = SCRIPT_DIR / "applets" / f"bm_{pipeline_name}"

    if workspace_dir.exists() and not replace:
        print(f"Workspace '{pipeline_name}' already exists. Use --replace to rebuild.")
        return

    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"{workspace_dir} directory created, now copying files")

    cli_dir = SCRIPT_DIR / "pipelineCLI"
    for file_path in cli_dir.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, workspace_dir / file_path.name)
            print(f"copied {file_path.name}")

    ui_dir = SCRIPT_DIR / "pipelineUI"
    for file_path in ui_dir.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, workspace_dir / file_path.name)
            print(f"copied {file_path.name}")

    user_dir = Path(directory)
    if user_dir.exists():
        for file_path in user_dir.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, workspace_dir / file_path.name)
                print(f"copied {file_path.name}")
    else:
        print(f"Warning: custom pipeline directory '{directory}' not found.")
        return

    print(f"Initialized workspace for pipeline '{pipeline_name}' at {workspace_dir}")


def new_custom_pipeline(pipeline_name, directory):
    workspace_dir = Path(directory) / pipeline_name
    if workspace_dir.exists():
        print(f"Workspace '{pipeline_name}' already exists.")
        return

    workspace_dir.mkdir(parents=True)
    print(f"{workspace_dir} directory created, now copying files")

    mock_dir = SCRIPT_DIR / "mock_pipeline"
    for file_path in mock_dir.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, workspace_dir / file_path.name)
            print(f"copied {file_path.name}")


def launch_pipeline(pipeline_name, stage, cli_or_ui):
    workspace_dir = SCRIPT_DIR / "applets" / f"bm_{pipeline_name}"
    filename = f"{stage}{cli_or_ui}.py"

    if stage not in ["initialise", "prepare", "train", "deploy"]:
        raise ValueError(
            "Stage name is invalid. Must be one of: prepare, train, deploy, all."
        )

    script_path = workspace_dir / filename
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find '{filename}' in {workspace_dir}")

    subprocess.run(["python", filename], cwd=workspace_dir)


def remove_pipeline(pipeline_name):
    workspace_dir = SCRIPT_DIR / "applets" / f"bm_{pipeline_name}"
    if workspace_dir.exists() and workspace_dir.is_dir():
        try:
            shutil.rmtree(workspace_dir)
            print(
                f"Removed workspace for pipeline '{pipeline_name}' at {workspace_dir}"
            )
        except PermissionError as e:
            print(f"⚠️ Could not delete '{workspace_dir}': {e}")
    else:
        print(f"Workspace '{pipeline_name}' does not exist.")


def list_pipelines(pipelines, builds):
    if builds:
        built_dir = SCRIPT_DIR / "applets"
        if built_dir.exists():
            print("Built pipelines are:")
            print(
                " |".join(
                    [
                        d.name[3:]
                        for d in built_dir.iterdir()
                        if d.is_dir() and d.name.startswith("bm_")
                    ]
                ),
                "|",
            )
    if pipelines:
        prebuilt_dir = SCRIPT_DIR / "pbm"
        if prebuilt_dir.exists():
            print("Pre-built pipelines are:")
            print(
                " |".join(
                    [
                        d.name[4:]
                        for d in prebuilt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("pbm_")
                    ]
                ),
                "|",
            )

def modify_pipeline(pipeline_name, directory_name):
    new_dir = Path(directory_name).resolve()

    if not new_dir.exists() or not new_dir.is_dir():
        # returns a message that directory_name wasn't found
        print(f"Directory '{directory_name}' not found")
        return
    else:
        moved_files = []
        
        for path in Path(f"{SCRIPT_DIR}/pbm/pbm_{pipeline_name}").iterdir():
            if path.is_file():
                dest = new_dir / path.name
                shutil.copy(str(path), str(dest))
                moved_files.append(dest)
        
        if pipeline_name:
            # prints ;he files that were copied to directory_name and 
            # a message that the checkout was completed successfully
            for file_path in moved_files:
                print(f"- {file_path.name} moved successfully")
            print(f"All files for {pipeline_name} were checked out successfully to '{directory_name}' for you to modify.")
        else:
            # returns a message that the checkout was completed
            return f"All files for {pipeline_name} were checked out successfully to '{directory_name}' for you to modify."   

def main():
    parser = argparse.ArgumentParser(description="BojAI Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_start = subparsers.add_parser("start", help="Start a pipeline")
    parser_start.add_argument("--pipeline", required=True)
    parser_start.add_argument("--stage", default="initialise")
    parser_start.add_argument("--ui", action="store_true")

    parser_build = subparsers.add_parser("build", help="Build a pipeline")
    parser_build.add_argument("--pipeline", required=True)
    parser_build.add_argument(
        "--replace", action="store_true", help="Overwrite existing build"
    )
    parser_build.add_argument(
        "--directory", default="none", help="Custom pipeline directory"
    )

    parser_remove = subparsers.add_parser("remove", help="Remove a built pipeline")
    parser_remove.add_argument("--pipeline", required=True)

    parser_create = subparsers.add_parser("create", help="Create a new custom pipeline")
    parser_create.add_argument("--pipeline", required=True)
    parser_create.add_argument("--directory", required=True)

    parser_list = subparsers.add_parser("list", help="List available/built pipelines")
    parser_list.add_argument("--pipelines", action="store_true")
    parser_list.add_argument("--builds", action="store_true")

    parser_modify = subparsers.add_parser('modify', help='Modify an existing pipeline')
    parser_modify.add_argument('--pipeline', required=True, help='Name of the pipeline to modify')
    parser_modify.add_argument('--directory', required=True, help='Directory to which the pipeline will be copied')

    args = parser.parse_args()

    if args.command == "start":
        launch_pipeline(args.pipeline, args.stage, "UI" if args.ui else "CLI")
    elif args.command == "build":
        if args.directory == "none":
            build_pipeline(args.pipeline, args.replace)
        else:
            build_dir(args.pipeline, args.directory, args.replace)
    elif args.command == "remove":
        remove_pipeline(args.pipeline)
    elif args.command == "create":
        new_custom_pipeline(args.pipeline, args.directory)
    elif args.command == "list":
        list_pipelines(args.pipelines, args.builds)
    elif args.command == 'modify':
        modify_pipeline(args.pipeline, args.directory)

if __name__ == "__main__":
    main()
