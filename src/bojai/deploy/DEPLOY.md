# Deploy

Deployment is the process of serving your trained BojAI pipeline as an API server. This allows you to make predictions using your model through HTTP requests.

## CLI Usage

To deploy a pipeline as an API server, use:

```bash
bojai deploy start <pipeline> <model_path> [--host HOST] [--port PORT]
```
- `<pipeline>`: Pipeline type to deploy. Must be one of: "CLI", "CLN", or "CLN-ML" (positional argument)
- `<model_path>`: Path to your saved model file (.bin) (positional argument)
- `--host`: Host address to bind the server to (default: "127.0.0.1")
- `--port`: Port number to bind the server to (default: 8000)

Other management commands:
- `bojai deploy stop <pipeline>` — Stop a running pipeline API server
- `bojai deploy status [pipeline]` — Get status of deployed pipelines


## API Endpoints

### POST /{your_chosen_pipeline_name}/predict
Makes predictions using the deployed model.

#### Request Format
- **CLI Pipeline**
  ```json
  {
    "input_data": {
      "image_path": "path/to/image.jpg"
      // or
      "image_data": "<base64-encoded-image-bytes>"
    }
  }
  ```
- **CLN Pipeline**
  ```json
  {
    "input_data": {
      "values": [1.0, 2.0, 3.0]
    }
  }
  ```
- **CLN-ML Pipeline**
  ```json
  {
    "input_data": {
      "values": [1.0, 2.0, 3.0]
    }
  }
  ```

#### Response Format
- **All Pipelines**
  ```json
  {
    "prediction": <predicted_value>,
    "confidence": <confidence_score>
  }
  ```

#### Response Codes
- **200 OK**: Prediction was successful and the response contains the prediction and confidence.
- **400 Bad Request**: The input data is missing required fields or is in the wrong format (e.g., missing 'image_path' for CLI, or 'values' for CLN/CLN-ML).
- **404 Not Found**: The model file could not be found at the specified path.
- **500 Internal Server Error**: An unexpected error occurred during prediction or model loading.

#### Examples
- **CLI Example**
  ```json
  {
    "input_data": {"image_path": "cat.jpg"}
  }
  // Response
  {
    "prediction": 1,
    "confidence": 0.95
  }
  ```
- **CLN Example**
  ```json
  {
    "input_data": {"values": [1.0, 2.0, 3.0]}
  }
  // Response
  {
    "prediction": 0,
    "confidence": 0.87
  }
  ```
- **CLN-ML Example**
  ```json
  {
    "input_data": {"values": [4.2, 5.1, 6.3]}
  }
  // Response
  {
    "prediction": 1,
    "confidence": 0.91
  }
  ```

### GET /placeholder/health
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
  "last_prediction_time": 1719123456,
  "total_predictions": 100,
  "average_prediction_time": 0.05,
  "error_count": 0,
  "model_info": {
    "type": "UserCLI",
    "input_size": [3, 500, 500],
    "output_size": 5
  }
}
```

#### Response Codes
- **200 OK**: Health check was successful and the server is running.

#### Field Descriptions
- **status**: Server health status ("healthy" or "unhealthy").
- **pipeline_type**: The type of pipeline deployed ("CLI", "CLN", or "CLN-ML").
- **model_loaded**: Whether the model is loaded and ready for inference.
- **uptime_seconds**: How long the server has been running (in seconds).
- **memory_usage_mb**: Current memory usage of the server (in megabytes).
- **cpu_usage_percent**: Current CPU usage percentage.
- **last_prediction_time**: Timestamp of the last prediction (or null if none yet).
- **total_predictions**: Total number of predictions served since startup.
- **average_prediction_time**: Average time (in seconds) taken per prediction.
- **error_count**: Number of errors encountered since startup.
- **model_info**: Basic information about the loaded model (type, input/output size).

## Error Handling

The server returns standardized HTTP error responses:

- **400 Bad Request** — Invalid input data (missing or wrong fields).
- **404 Not Found** — Model file not found at the specified path.
- **500 Internal Server Error** — Server-side errors (e.g., model loading or inference failure).

## Usage Examples

### Starting a Server
```bash
# Deploy a CLI pipeline
bojai deploy start cli /path/to/model.bin

# Deploy a CLN pipeline with custom host and port
bojai deploy start cln /path/to/model.bin --host localhost --port 8080
```

### Making Predictions
```python
import requests

# CLI pipeline prediction
response = requests.post(
    "http://localhost:8000/placeholder/predict",
    json={"input_data": {"image_path": "cat.jpg"}}
)
print(response.json())

# CLN pipeline prediction
response = requests.post(
    "http://localhost:8000/placeholder/predict",
    json={"input_data": {"values": [1.0, 2.0, 3.0]}}
)
print(response.json())
```

### Checking Health
```python
import requests

response = requests.get("http://localhost:8000/placeholder/health")
print(response.json())
```


Updated on 3rd July 2025

