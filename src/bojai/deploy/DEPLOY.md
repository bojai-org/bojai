# Deploy

Deployment is the process of serving your trained BojAI pipeline as an API server. This allows you to make predictions using your model through HTTP requests.

## Sub-commands

* **--pipeline** — Select the pipeline type you want to deploy. Must be one of: "CLI", "CLN", or "CLN-ML"
* **--model-path** — Path to your saved model file (.bin)
* **--host** — Host address to bind the server to (default: "0.0.0.0")
* **--port** — Port number to bind the server to (default: 8000)

## API Endpoints

### POST /predict
Makes predictions using the deployed model.

#### Request Format
```json
{
    "input_data": {
        // Pipeline-specific input data
    }
}
```

#### Response Format
```json
{
    "prediction": "prediction_result",
    "confidence": 0.95,
    "metadata": {
        // Additional prediction metadata
    }
}
```

### GET /health
Checks the health and status of the deployed server.

#### Response Format
```json
{
    "status": "healthy",
    "pipeline_type": "CLI",
    "model_loaded": true,
    "uptime_seconds": 3600,
    "memory_usage_mb": 512,
    "cpu_usage_percent": 25,
    "last_prediction_time": 1619123456,
    "total_predictions": 100
}
```

## Pipeline Types

### CLI
Image-based classification pipeline.
* Input: Image data
* Output: Classification prediction

### CLN
Data-based classification pipeline.
* Input: Structured data
* Output: Classification prediction

### CLN-ML
Advanced ML classification pipeline.
* Input: Structured data
* Output: Classification prediction with confidence scores

## Error Handling

The server returns standardized HTTP error responses:

* **400 Bad Request** — Invalid input data
* **404 Not Found** — Model file not found
* **500 Internal Server Error** — Server-side errors

## Examples

### Starting a Server
```bash
# Deploy a CLI pipeline
bojai deploy --pipeline CLI --model-path /path/to/model.bin

# Deploy a CLN pipeline with custom host and port
bojai deploy --pipeline CLN --model-path /path/to/model.bin --host localhost --port 8080
```

### Making Predictions
```python
import requests

# CLI pipeline prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "input_data": {
            "image": "base64_encoded_image"
        }
    }
)

# CLN pipeline prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "input_data": {
            "features": [1.0, 2.0, 3.0]
        }
    }
)
```

### Checking Health
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```


Updated on 10 May 2025

