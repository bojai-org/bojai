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

def build_pipeline(pipeline_name, directory_arg, replace):
    workspace_dir = SCRIPT_DIR / "applets" / f"bm_{pipeline_name}"

    if workspace_dir.exists() and not replace:
        print(f"Workspace '{pipeline_name}' already exists. Use --replace to rebuild.")
        return

    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"{workspace_dir} directory created.")

    # Step 1: Copy all files from user directories (priority)
    if directory_arg.lower() != "none":
        directories = [Path(d.strip()) for d in directory_arg.split(",")]
        for user_dir in directories:
            if user_dir.exists():
                for file_path in user_dir.iterdir():
                    if file_path.is_file():
                        shutil.copy(file_path, workspace_dir / file_path.name)
                        print(f"[user] copied {file_path.name}")
            else:
                print(f"Warning: custom pipeline directory '{user_dir}' not found.")

    # Step 2: Copy pipeline-specific files
    specific_dir = SCRIPT_DIR / "pbm" / f"pbm_{pipeline_name}"
    if specific_dir.exists():
        for file_path in specific_dir.iterdir():
            if file_path.is_file():
                dest_file = workspace_dir / file_path.name
                if not dest_file.exists():
                    shutil.copy(file_path, dest_file)
                    print(f"[specific] copied {file_path.name}")
                else:
                    print(f"[specific] skipped {file_path.name} (already exists)")
    
    # Step 3: Copy from default CLI/UI folders only if file not already copied
    for folder in ["pipelineCLI", "pipelineUI"]:
        source_dir = SCRIPT_DIR / folder
        for file_path in source_dir.iterdir():
            if file_path.is_file():
                dest_file = workspace_dir / file_path.name
                if not dest_file.exists():
                    shutil.copy(file_path, dest_file)
                    print(f"[default] copied {file_path.name}")
                else:
                    print(f"[default] skipped {file_path.name} (already exists)")
    

    print(f"✅ Initialized workspace for pipeline '{pipeline_name}' at {workspace_dir}")


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
    
def checkout_directory(directory_name, cli_or_ui) :
    new_dir = Path(directory_name).resolve()

    if not new_dir.exists() or not new_dir.is_dir():
        # returns a message that directory_name wasn't found
        print(f"Directory '{directory_name}' not found")
        return
    
    else:
        moved_files = []
        
        for path in Path(f"{SCRIPT_DIR}/pipeline{cli_or_ui}").iterdir():
            if path.is_file():
                dest = new_dir / path.name
                shutil.copy(str(path), str(dest))
                moved_files.append(dest)
        
        if cli_or_ui:
            # prints ;he files that were copied to directory_name and 
            # a message that the checkout was completed successfully
            for file_path in moved_files:
                print(f"- {file_path.name} moved successfully")
            print(f"\nAll files in working directory were SUCCESSFULLY checked out into '{directory_name}'")
        else:
            # returns a message that the checkout was completed
            return f"All files were checked out successfully to '{directory_name}'"           


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
    parser_build.add_argument("--directory", default="none", help="Comma-separated custom pipeline directories")

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

    # Modify command
    parser_modify = subparsers.add_parser('modify', help='Modify an existing pipeline')
    parser_modify.add_argument('--pipeline', required=True, help='Name of the pipeline to modify')
    parser_modify.add_argument('--directory', required=True, help='Directory to which the pipeline will be copied')

    # Checkout command
    parser_checkout = subparsers.add_parser("checkout", help = "Checkout an existing directory")
    parser_checkout.add_argument("--directory", required = True)
    parser_checkout.add_argument("--ui", action = "store_true")

    args = parser.parse_args()

    if args.command == "start":
        launch_pipeline(args.pipeline, args.stage, "UI" if args.ui else "CLI")
    elif args.command == "build":
        build_pipeline(args.pipeline, args.directory, args.replace)
    elif args.command == "remove":
        remove_pipeline(args.pipeline)
    elif args.command == "create":
        new_custom_pipeline(args.pipeline, args.directory)
    elif args.command == "list":
        list_pipelines(args.pipelines, args.builds)
    elif args.command == 'modify':
        modify_pipeline(args.pipeline, args.directory)
    elif args.command == "checkout":
        checkout_directory(args.directory, "UI" if args.ui else "CLI")

if __name__ == "__main__":
    main()