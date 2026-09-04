# ==============================================================================
# TERRA-SR — Production Dockerfile
# AI Multispectral Super-Resolution & Geo-Accurate Satellite Intelligence Platform
# ==============================================================================

FROM python:3.11-slim

# Set environment variables for Python runtime optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Install runtime system libraries (OpenMP for PyTorch, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY configs/ ./configs/
COPY data/processed/ ./data/processed/

# Ensure output and static caching directories exist
RUN mkdir -p /app/outputs /app/scratch /app/app/static

# Expose HTTP service port
EXPOSE 8080

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start TERRA-SR production server
CMD ["python", "app/satellite_enhancer.py"]
