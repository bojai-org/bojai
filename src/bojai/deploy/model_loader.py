import torch
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException

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

    def load_model(self, model_path: str) -> Any:
        self.validate_model_path(model_path)
        try:
            # Placeholder: In the future, extract pipeline_name from the model file here
            model = torch.load(model_path, map_location="cpu")
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail={
                "error": "Failed to load model",
                "reason": str(e)
            })

    def validate_model_path(self, model_path: str):
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail={
                "error": "Model file not found",
                "path": model_path
            })
        if not model_path.endswith('.bin'):
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