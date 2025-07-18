import torch
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException
from .logging_utils import get_logger
logger = get_logger(__name__)

class ModelLoader:
    """
    Model loader class for handling different pipeline types during prediction.
    
    Attributes:
        pipeline_type: Type of pipeline (CLI, CLN, CLN-ML)
    """
    
    def __init__(self, pipeline_type: str):
        if pipeline_type not in ["CLI", "CLN", "CLN-ML"]:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
        self.pipeline_type = pipeline_type
        self.model_path = None

    def load_model(self, model_path: str) -> Any:
        self.validate_model_path(model_path)
        self.model_path = model_path
        try:
            logger.info(f"Loading model from {model_path}")
            model = torch.load(model_path, map_location="cpu")
            if hasattr(model, 'eval'):
                model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load model from {model_path}: {e}")
            raise HTTPException(status_code=500, detail={
                "error": "Failed to load model",
                "reason": str(e)
            })

    def validate_model_path(self, model_path: str):
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(status_code=404, detail={
                "error": "Model file not found",
                "path": model_path
            })
        if not model_path.endswith('.bin'):
            logger.error(f"Invalid model file extension: {model_path}")
            raise HTTPException(status_code=400, detail={
                "error": "Invalid model file extension",
                "expected": ".bin",
                "received": model_path
            })

    def get_model_config(self) -> Dict[str, Any]:
        if self.pipeline_type == "CLI":
            return {"input_size": (3, 500, 500), "output_size": 5}
        elif self.pipeline_type == "CLN":
            return {"input_size": "dynamic", "output_size": 1}
        elif self.pipeline_type == "CLN-ML":
            return {"input_size": "dynamic", "output_size": 1}
        else:
            return {}  # Should not happen 