import argparse
import shutil
import subprocess
from pathlib import Path
import logging

# Base directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            

def main():
    parser = argparse.ArgumentParser(description="BojAI Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    parser_start = subparsers.add_parser("start", help="Start a pipeline")
    parser_start.add_argument("--pipeline", required=True)
    parser_start.add_argument("--stage", default="initialise")
    parser_start.add_argument("--ui", action="store_true")

    # Build command
    parser_build = subparsers.add_parser("build", help="Build a pipeline")
    parser_build.add_argument("--pipeline", required=True)
    parser_build.add_argument("--replace", action="store_true", help="Overwrite existing build")
    parser_build.add_argument("--directory", default="none", help="Custom pipeline directory")

    # Remove command
    parser_remove = subparsers.add_parser("remove", help="Remove a built pipeline")
    parser_remove.add_argument("--pipeline", required=True)

    # Create command
    parser_create = subparsers.add_parser("create", help="Create a new custom pipeline")
    parser_create.add_argument("--pipeline", required=True)
    parser_create.add_argument("--directory", required=True)

    # List command
    parser_list = subparsers.add_parser("list", help="List available/built pipelines")
    parser_list.add_argument("--pipelines", action="store_true")
    parser_list.add_argument("--builds", action="store_true")

    # Deploy command
    parser_deploy = subparsers.add_parser("deploy", help="Deploy a pipeline as an API")
    deploy_subparsers = parser_deploy.add_subparsers(dest="deploy_command", required=True)

    # Deploy start
    deploy_start = deploy_subparsers.add_parser("start", help="Start a pipeline API server")
    deploy_start.add_argument("pipeline", help="Name of the pipeline to start")
    deploy_start.add_argument("model_path", help="Path to the trained model file (.bin)")
    deploy_start.add_argument("--port", "-p", type=int, default=8000, help="Port to run the API server on")
    deploy_start.add_argument("--host", default="127.0.0.1", help="Host to run the API server on")

    # Deploy stop
    deploy_stop = deploy_subparsers.add_parser("stop", help="Stop a running pipeline API server")
    deploy_stop.add_argument("pipeline", help="Name of the pipeline to stop")

    # Deploy status
    deploy_status = deploy_subparsers.add_parser("status", help="Get status of deployed pipelines")
    deploy_status.add_argument("pipeline", nargs="?", help="Name of the pipeline to check status")

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
    elif args.command == "deploy":
        from bojai.deploy.cli import start_pipeline, stop_pipeline, get_pipeline_status
        
        if args.deploy_command == "start":
            start_pipeline(args)
        elif args.deploy_command == "stop":
            stop_pipeline(args)
        elif args.deploy_command == "status":
            get_pipeline_status(args)

if __name__ == "__main__":
    main()