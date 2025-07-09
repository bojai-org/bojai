import signal
import subprocess
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os

# Global variable to store running processes
running_processes = {}

class PipelineRequest(BaseModel):
    input_data: dict

def start_pipeline(args):
    """
    Starts a pipeline as an API server.
    Args:
        args: Command line arguments containing pipeline type, model path, port, and host
    Note: For now, uses 'placeholder' as the pipeline name for endpoints.
    In the future, pipeline name could be extracted from the .bin file or passed as an argument.
    """
    pipeline_type = args.pipeline
    model_path = args.model_path
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 8000)
    # Start the server as a subprocess
    process = subprocess.Popen([
        sys.executable, '-m', 'bojai.deploy.server',
        '--pipeline-type', pipeline_type,
        '--model-path', model_path,
        '--host', str(host),
        '--port', str(port)
    ])
    running_processes[pipeline_type] = process.pid
    print(f"Started {pipeline_type} pipeline server at {host}:{port} (PID: {process.pid})")

def stop_pipeline(args):
    """
    Stops a running pipeline API server.
    Args:
        args: Command line arguments containing pipeline type
    """
    pipeline_type = args.pipeline
    pid = running_processes.get(pipeline_type)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped {pipeline_type} pipeline server (PID: {pid})")
            del running_processes[pipeline_type]
        except Exception as e:
            print(f"Failed to stop server: {e}")
    else:
        print(f"No running server found for pipeline: {pipeline_type}")

def get_pipeline_status(args):
    """
    Gets the status of deployed pipelines.
    Args:
        args: Command line arguments containing optional pipeline type
    """
    if hasattr(args, 'pipeline') and args.pipeline:
        pid = running_processes.get(args.pipeline)
        if pid:
            print(f"Pipeline {args.pipeline} is running (PID: {pid})")
        else:
            print(f"Pipeline {args.pipeline} is not running.")
    else:
        if running_processes:
            for name, pid in running_processes.items():
                print(f"Pipeline {name} is running (PID: {pid})")
        else:
            print("No pipelines are currently running.") 