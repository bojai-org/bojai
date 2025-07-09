# BojAI Deployment Documentation

BojAI provides a powerful deployment system that allows you to serve your trained machine learning models as REST APIs. This documentation covers everything you need to know to deploy and use BojAI pipelines in production.

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Types](#pipeline-types)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

Deploy your first model in under 5 minutes:

```bash
# 1. Install BojAI
pip install bojai

# 2. Deploy a model
bojai deploy start cln-ml my_model.bin --port 8080

# 3. Test the API
curl http://localhost:8080/ping
```

## Pipeline Types

BojAI deploy supports three main pipeline types:

### 1. CLI (Computer Vision)
- **Purpose**: Image classification and computer vision tasks
- **Input**: Images (file paths or base64 encoded)
- **Output**: Classification predictions with confidence scores
- **Use Cases**: Object detection, image classification, visual analysis

### 2. CLN (Classification)
- **Purpose**: Numerical data classification
- **Input**: Numerical arrays or lists
- **Output**: Binary or multi-class predictions
- **Use Cases**: Risk assessment, fraud detection, customer segmentation

### 3. CLN-ML (Advanced ML)
- **Purpose**: Advanced machine learning with metadata
- **Input**: Numerical arrays with optional parameters
- **Output**: Predictions with confidence and metadata
- **Use Cases**: Complex ML workflows, feature importance analysis

## Deployment

### Command Line Interface

#### Start a Pipeline Server

```bash
bojai deploy start <pipeline-type> <model-path> [options]
```

**Parameters:**
- `pipeline-type`: One of `cli`, `cln`, or `cln-ml` (case-insensitive)
- `model-path`: Path to your trained model file (.bin format)

**Options:**
- `--port`: Port number (default: 8000)
- `--host`: Host address (default: 127.0.0.1)

**Examples:**

```bash
# Deploy a computer vision model
bojai deploy start cli vision_model.bin --port 8080

# Deploy a classification model
bojai deploy start cln classifier.bin --host 0.0.0.0 --port 9000

# Deploy an advanced ML model
bojai deploy start cln-ml advanced_model.bin --port 8080
```

#### Manage Running Pipelines

```bash
# Stop a pipeline
bojai deploy stop <pipeline-name>

# Check pipeline status
bojai deploy status [pipeline-name]

# List all running pipelines
bojai deploy status
```

### Model File Format

BojAI expects PyTorch model files in `.bin` format:

```python
# Save your model
import torch

# For a complete model
torch.save(model, 'my_model.bin')

# For model state dict
torch.save(model.state_dict(), 'my_model.bin')
```

## API Reference

### Base URL
All endpoints are available at: `http://your-server:port`


### Endpoints

#### 1. Health Check
**GET** `/ping`

Check if the server is running.

**Response:**
```json
{
  "status": "ok"
}
```

#### 2. Model Information
**GET** `/{model_name}/check_model`

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_type": "OrderedDict",
  "model_path": "/path/to/model.bin",
  "pipeline_type": "CLN-ML",
  "is_callable": true,
  "model_attributes": ["layer1", "layer2", "output"],
  "model_size": 3,
  "device": "cpu"
}
```

#### 3. Health Monitoring
**GET** `/{model_name}/health`

Get comprehensive server and model health information.

**Response:**
```json
{
  "status": "healthy",
  "pipeline_type": "CLN-ML",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "memory_usage_mb": 512.3,
  "cpu_usage_percent": 15.2,
  "last_prediction_time": 1640995200.0,
  "total_predictions": 150,
  "average_prediction_time": 0.045,
  "error_count": 2,
  "model_info": {
    "type": "MyModel",
    "input_size": "dynamic",
    "output_size": 1
  }
}
```

#### 4. Predictions

##### CLI Pipeline (Images)
**POST** `/{model_name}/predict`

**Request Body:**
```json
{
  "input_data": {
    "image_path": "/path/to/image.jpg"
  }
}
```

Or with base64 image:
```json
{
  "input_data": {
    "image_data": "base64_encoded_image_string"
  }
}
```

**Response:**
```json
{
  "prediction": 2,
  "confidence": 0.95
}
```

##### CLN/CLN-ML Pipeline (Numerical Data)
**POST** `/{model_name}/predict`

**Request Body:**
```json
{
  "input_data": {
    "values": [1.0, 2.5, 3.2, 0.8]
  }
}
```

**Response (CLN):**
```json
{
  "prediction": 1,
  "confidence": 0.87
}
```

**Response (CLN-ML):**
```json
{
  "prediction": 1,
  "confidence": 0.87,
  "metadata": {
    "processing_time": 0.045,
    "feature_importance": [0.3, 0.4, 0.2, 0.1]
  }
}
```

### Error Responses

All endpoints return appropriate HTTP status codes:

- **200 OK**: Successful operation
- **400 Bad Request**: Invalid input data
- **404 Not Found**: Model file or endpoint not found
- **500 Internal Server Error**: Server or model error

Error response format:
```json
{
  "detail": {
    "error": "Error description"
  }
}
```

## Examples

### Python Client

```python
import requests
import json

# Server configuration
BASE_URL = "http://localhost:8080"
MODEL_NAME = "my_model"

# Health check
response = requests.get(f"{BASE_URL}/ping")
print(f"Server status: {response.json()}")

# Check model
response = requests.get(f"{BASE_URL}/{MODEL_NAME}/check_model")
model_info = response.json()
print(f"Model type: {model_info['model_type']}")

# Make prediction (CLN-ML)
data = {
    "input_data": {
        "values": [1.0, 2.0, 3.0, 4.0]
    }
}
response = requests.post(f"{BASE_URL}/{MODEL_NAME}/predict", json=data)
result = response.json()
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/ping

# Check model
curl http://localhost:8080/my_model/check_model

# Make prediction
curl -X POST http://localhost:8080/my_model/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"values": [1.0, 2.0, 3.0]}}'

# Image prediction (CLI)
curl -X POST http://localhost:8080/vision_model/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image_path": "/path/to/image.jpg"}}'
```

## Monitoring and Logging

### Log Levels

BojAI uses structured logging with the following levels:
- **INFO**: General operational information
- **WARNING**: Potential issues
- **ERROR**: Errors that don't stop the service
- **CRITICAL**: Critical errors that may stop the service

### Metrics

The `/health` endpoint provides key metrics:
- Uptime
- Memory usage
- CPU usage
- Prediction count and timing
- Error count

### Integration with Monitoring Tools

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('bojai_predictions_total', 'Total predictions')
prediction_duration = Histogram('bojai_prediction_duration_seconds', 'Prediction duration')
```

## Troubleshooting

### Common Issues

#### 1. Model File Not Found
**Error**: `Model file not found: model.bin`
**Solution**: Check the file path and ensure the model file exists

#### 2. Invalid Pipeline Type
**Error**: `Unsupported pipeline type: invalid_type`
**Solution**: Use one of: `cli`, `cln`, or `cln-ml`

#### 3. Model Not Callable
**Error**: `Model is not callable - invalid model file`
**Solution**: Ensure your model file contains a proper PyTorch model, not just a state dict

#### 4. Port Already in Use
**Error**: `Address already in use`
**Solution**: Use a different port or stop the existing service

### Debug Mode

Enable debug logging:

```bash
export BOJAI_LOG_LEVEL=DEBUG
bojai deploy start cln-ml model.bin --port 8080
```

### Performance Optimization

1. **Use GPU**: Ensure PyTorch is installed with CUDA support
2. **Batch Processing**: Implement batch prediction endpoints for high throughput
3. **Model Optimization**: Use TorchScript or ONNX for faster inference
4. **Caching**: Implement prediction result caching for repeated inputs

