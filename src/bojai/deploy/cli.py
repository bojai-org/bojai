import signal
import subprocess
import shutil
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os
from .logging_utils import get_logger
logger = get_logger(__name__)

# Global variable to store running processes
running_processes = {}

class PipelineRequest(BaseModel):
    input_data: dict

def start_pipeline(args):
    pipeline_type = args.pipeline
    model_path = args.model_path
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 8000)
    provider = getattr(args, 'provider', 'localhost')

    if provider == 'azure':
        # Check for Docker
        if not shutil.which('docker'):
            logger.error('Docker is not installed or not in PATH.')
            raise RuntimeError('Docker is required for Azure deployment.')
        # Check for Azure CLI
        if not shutil.which('az'):
            logger.error('Azure CLI (az) is not installed or not in PATH.')
            raise RuntimeError('Azure CLI is required for Azure deployment.')
        # Check Azure login
        try:
            result = subprocess.run(['az', 'account', 'show'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error('You are not logged in to Azure CLI. Please run "az login".')
                print(result.stderr)
                raise RuntimeError('Azure CLI login required.')
        except Exception as e:
            logger.error(f'Azure CLI check failed: {e}')
            raise
        # Prompt for required Azure info if missing
        acr_name = getattr(args, 'acr_name', None) or input('Enter Azure Container Registry name: ')
        resource_group = getattr(args, 'resource_group', None) or input('Enter Azure Resource Group: ')
        region = getattr(args, 'region', None) or input('Enter Azure region (e.g. eastus): ')
        container_name = getattr(args, 'container_name', None) or input('Enter Azure Container Instance name: ')
        dns_name_label = getattr(args, 'dns_name_label', None) or input('Enter DNS name label for Azure Container Instance: ')
        # Build Docker image
        image_tag = f'{acr_name}.azurecr.io/bojai-model:latest'
        logger.info(f'Building Docker image: {image_tag}')
        build_cmd = [
            'docker', 'build', '-t', image_tag, '--build-arg', f'MODEL_PATH={model_path}', '.'
        ]
        result = subprocess.run(build_cmd)
        if result.returncode != 0:
            logger.error('Docker build failed.')
            raise RuntimeError('Docker build failed.')
        logger.info('Docker image built successfully.')
        # Azure CLI: login to ACR
        logger.info(f'Logging in to Azure Container Registry: {acr_name}')
        result = subprocess.run(['az', 'acr', 'login', '--name', acr_name])
        if result.returncode != 0:
            logger.error('Azure ACR login failed.')
            raise RuntimeError('Azure ACR login failed.')
        # Push Docker image to ACR
        logger.info(f'Pushing Docker image to ACR: {image_tag}')
        result = subprocess.run(['docker', 'push', image_tag])
        if result.returncode != 0:
            logger.error('Docker push to ACR failed.')
            raise RuntimeError('Docker push to ACR failed.')
        logger.info('Docker image pushed to ACR successfully.')
        # Deploy to Azure Container Instance
        logger.info(f'Deploying to Azure Container Instance: {container_name}')
        aci_cmd = [
            'az', 'container', 'create',
            '--resource-group', resource_group,
            '--name', container_name,
            '--image', image_tag,
            '--registry-login-server', f'{acr_name}.azurecr.io',
            '--dns-name-label', dns_name_label,
            '--ports', str(port),
            '--location', region
        ]
        result = subprocess.run(aci_cmd)
        if result.returncode != 0:
            logger.error('Azure Container Instance deployment failed.')
            raise RuntimeError('Azure Container Instance deployment failed.')
        logger.info('Deployment to Azure Container Instance successful.')
        print(f'Your API should be available at: http://{dns_name_label}.{region}.azurecontainer.io:{port}/ping')
        return
    else:
        logger.info(f"Starting {pipeline_type} pipeline server at {host}:{port}")
        try:
            from bojai.deploy.server import PipelineServer
            server = PipelineServer(pipeline_type, model_path)
            logger.info(f"Server initialized successfully, starting on {host}:{port}")
            server.start(host, port)
        except Exception as e:
            logger.exception(f"Failed to start server: {e}")
            raise RuntimeError(f"Server failed to start: {e}")

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
            logger.info(f"Stopped {pipeline_type} pipeline server (PID: {pid})")
            del running_processes[pipeline_type]
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
    else:
        logger.warning(f"No running server found for pipeline: {pipeline_type}")

def get_pipeline_status(args):
    """
    Gets the status of deployed pipelines.
    Args:
        args: Command line arguments containing optional pipeline type
    """
    if hasattr(args, 'pipeline') and args.pipeline:
        pid = running_processes.get(args.pipeline)
        if pid:
            logger.info(f"Pipeline {args.pipeline} is running (PID: {pid})")
        else:
            logger.info(f"Pipeline {args.pipeline} is not running.")
    else:
        if running_processes:
            for name, pid in running_processes.items():
                logger.info(f"Pipeline {name} is running (PID: {pid})")
        else:
            logger.info("No pipelines are currently running.") 