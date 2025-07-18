"""
FastAPI server implementation for BojAI pipeline deployment.
Handles model loading, inference, and API endpoints.
"""

from fastapi import FastAPI, HTTPException
import torch
import psutil
import time
from typing import Dict, Any

# Import BojAI components
from bojai.pipelineCLI.prepare import Prepare
from bojai.deploy.model_loader import ModelLoader
from bojai.deploy.user import UserCLI, UserCLN, UserCLNML
from bojai.deploy.models import PipelineRequest, PipelineResponse, HealthResponse
from .logging_utils import get_logger
logger = get_logger(__name__)

class PipelineServer:
    """
    Server class for handling pipeline deployment.
    """
    def __init__(self, pipeline_type: str, model_path: str):
        # Convert to uppercase to handle case-insensitive input
        pipeline_type = pipeline_type.upper()
        if pipeline_type not in ["CLI", "CLN", "CLN-ML"]:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
        self.pipeline_type = pipeline_type
        self.hyper_params = self._get_hyper_params()
        self.model_loader = ModelLoader(pipeline_type)
        try:
            self.pipeline = self.model_loader.load_model(model_path)
        except Exception as e:
            logger.exception(f"Failed to initialize pipeline: {str(e)}")
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")
        self.user = self.setup_user()
        logger.info(f"Initialized PipelineServer for {pipeline_type} with model {model_path}")
        # Extract pipeline name from model file or use default
        self.pipeline_name = self._extract_pipeline_name(model_path)
        self.app = self.create_app()
        self.start_time = time.time()
        self.last_prediction = None
        self.prediction_count = 0
        self.error_count = 0
        self.total_prediction_time = 0.0

    def _get_hyper_params(self) -> Dict[str, Any]:
        if self.pipeline_type == "CLI":
            return {"input_size": (3, 500, 500), "output_size": 5}
        elif self.pipeline_type == "CLN":
            return {"hidden_size": 256, "num_layers": 3, "num_batches": 64}
        elif self.pipeline_type == "CLN-ML":
            return {"hidden_size": 256, "num_layers": 3, "num_batches": 64}
        else:
            return {}

    def _extract_pipeline_name(self, model_path: str) -> str:
        """Extract pipeline name from model file path"""
        from pathlib import Path
        model_name = Path(model_path).stem
        return model_name if model_name else "default"

    def setup_user(self):
        if self.pipeline_type == "CLI":
            return UserCLI(self.pipeline, device="cpu")
        elif self.pipeline_type == "CLN":
            return UserCLN(self.pipeline, device="cpu")
        elif self.pipeline_type == "CLN-ML":
            return UserCLNML(self.pipeline, device="cpu")
        else:
            raise RuntimeError("Unsupported pipeline type for user setup")

    def create_app(self):
        app = FastAPI(title=f"BojAI {self.pipeline_type} Pipeline Server",
                      description="API server for BojAI pipeline deployment.")
        self.add_routes(app)
        return app

    def add_routes(self, app: FastAPI):
        pipeline_prefix = f"/{self.pipeline_name}"

        @app.get("/ping")
        async def ping():
            return {"status": "ok"}

        @app.get(f"{pipeline_prefix}/check_model")
        async def check_model():
            """Check model properties and contents"""
            try:
                model_info = {
                    "model_type": type(self.pipeline).__name__,
                    "model_path": self.model_loader.model_path if hasattr(self.model_loader, 'model_path') else "unknown",
                    "pipeline_type": self.pipeline_type,
                    "is_callable": callable(self.pipeline),
                    "model_attributes": list(self.pipeline.keys()) if hasattr(self.pipeline, 'keys') else [],
                    "model_size": len(self.pipeline) if hasattr(self.pipeline, '__len__') else "unknown",
                    "device": self.device if hasattr(self, 'device') else "cpu"
                }
                return model_info
            except Exception as e:
                logger.error(f"Error checking model: {e}")
                raise HTTPException(status_code=500, detail={"error": f"Failed to check model: {str(e)}"})

        @app.post(f"{pipeline_prefix}/predict", response_model=PipelineResponse)
        async def predict(request: PipelineRequest):
            start_pred = time.time()
            try:
                input_data = request.input_data
                # For CLI: expects 'image_path' or 'image_data' (bytes)
                # For CLN/CLN-ML: expects 'values' (list)
                if self.pipeline_type == "CLI":
                    if "image_path" in input_data:
                        result = self.user.use_model(input_data["image_path"])
                    elif "image_data" in input_data:
                        result = self.user.use_model(input_data["image_data"])
                    else:
                        raise HTTPException(status_code=400, detail={
                            "error": "Missing image_path or image_data for CLI pipeline"
                        })
                elif self.pipeline_type in ["CLN", "CLN-ML"]:
                    if "values" in input_data:
                        result = self.user.use_model(input_data["values"])
                    else:
                        raise HTTPException(status_code=400, detail={
                            "error": "Missing 'values' for CLN/CLN-ML pipeline"
                        })
                else:
                    raise HTTPException(status_code=400, detail={
                        "error": "Unsupported pipeline type"
                    })
                self.prediction_count += 1
                self.last_prediction = time.time()
                pred_time = self.last_prediction - start_pred
                self.total_prediction_time += pred_time
                return PipelineResponse(
                    prediction=result.get("prediction"),
                    confidence=result.get("confidence")
                )
            except Exception as e:
                self.error_count += 1
                raise HTTPException(status_code=500, detail={"error": str(e)})

        @app.get(f"{pipeline_prefix}/health", response_model=HealthResponse)
        async def health():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            uptime = time.time() - self.start_time
            avg_pred_time = (self.total_prediction_time / self.prediction_count) if self.prediction_count else 0.0
            return HealthResponse(
                status="healthy",
                pipeline_type=self.pipeline_type,
                model_loaded=True,
                uptime_seconds=uptime,
                memory_usage_mb=mem.used / (1024 * 1024),
                cpu_usage_percent=cpu,
                last_prediction_time=self.last_prediction,
                total_predictions=self.prediction_count,
                average_prediction_time=avg_pred_time,
                error_count=self.error_count,
                model_info={
                    "type": type(self.pipeline).__name__,
                    "input_size": self.hyper_params.get("input_size"),
                    "output_size": self.hyper_params.get("output_size")
                }
            )

    def start(self, host: str, port: int):
        import uvicorn

        if self.pipeline is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded")
        logger.info(f"Starting Uvicorn server at {host}:{port}")
        uvicorn.run(self.app, host=host, port=port) 