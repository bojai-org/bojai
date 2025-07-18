#!/usr/bin/env python3
"""
Entry point for bojai.deploy.server module.
This allows the server to be started as a subprocess using:
python -m bojai.deploy.server --pipeline-type <type> --model-path <path> --host <host> --port <port>
"""

import argparse
import sys
from .server import PipelineServer

def main():
    parser = argparse.ArgumentParser(description="BojAI Pipeline Server")
    parser.add_argument("--pipeline-type", required=True, 
                       choices=["CLI", "CLN", "CLN-ML"],
                       help="Type of pipeline to deploy")
    parser.add_argument("--model-path", required=True,
                       help="Path to the model file (.bin)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the server to")
    
    args = parser.parse_args()
    
    try:
        # Create and start the server
        server = PipelineServer(args.pipeline_type, args.model_path)
        server.start(args.host, args.port)
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 