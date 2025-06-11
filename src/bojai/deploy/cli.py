import signal
import subprocess
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Global variable to store running processes
running_processes = {}

class PipelineRequest(BaseModel):
    input_data: dict

def start_pipeline(args):
    """
    Starts a pipeline as an API server.
    
    Args:
        args: Command line arguments containing pipeline name, model path, port, and host
    
    Logic:
    1. Validate model path and pipeline type
    2. Create FastAPI app
    3. Load model
    4. Start server
    5. Store process info
    """
    # Pseudo-code:
    # validate_pipeline_type(args.pipeline)
    # validate_model_path(args.model_path)
    # app = create_fastapi_app(args.pipeline)
    # model = load_model(args.model_path)
    # process = start_server(app, args.host, args.port)
    # store_process_info(args.pipeline, process)

def stop_pipeline(args):
    """
    Stops a running pipeline API server.
    
    Args:
        args: Command line arguments containing pipeline name
    
    Logic:
    1. Find process by pipeline name
    2. Terminate process
    3. Remove from running processes
    """
    # Pseudo-code:
    # process = get_process_by_name(args.pipeline)
    # terminate_process(process)
    # remove_from_running_processes(args.pipeline)

def get_pipeline_status(args):
    """
    Gets the status of deployed pipelines.
    
    Args:
        args: Command line arguments containing optional pipeline name
    
    Logic:
    1. If pipeline specified, get its status
    2. Otherwise, list all running pipelines
    """
    # Pseudo-code:
    # if args.pipeline:
    #     return get_single_pipeline_status(args.pipeline)
    # else:
    #     return list_all_running_pipelines() 