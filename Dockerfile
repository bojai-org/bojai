# BojAI API Dockerfile
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the BojAI codebase (only src/ and needed files)
COPY src/ ./src/
COPY README.md ./
COPY LICENSE ./

# Set workdir to src for module imports
WORKDIR /app/src

# Expose the API port
EXPOSE 8080

# Default command (can be overridden)
CMD ["python", "-m", "bojai.deploy.server", "--pipeline-type", "CLN-ML", "--model-path", "my_model.bin", "--host", "0.0.0.0", "--port", "8080"] 