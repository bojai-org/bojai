"""
Data models for API request/response handling.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional

class PipelineRequest(BaseModel):
    """Request model for pipeline predictions"""
    input_data: Dict[str, Any]

class PipelineResponse(BaseModel):
    """Response model for pipeline predictions"""
    prediction: Any
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Response model for health checks"""
    status: str
    pipeline_type: str
    model_loaded: bool
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_prediction_time: Optional[float] = None
    total_predictions: int = 0 