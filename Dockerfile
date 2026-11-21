# Production Dockerfile for ChurnSense
# Single-container architecture serving FastAPI backend and premium frontend

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Install system dependencies (libgomp1 for XGBoost/OpenMP support, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules, pre-trained artifacts, and web assets
COPY src/ ./src/
COPY models/ ./models/
COPY web/ ./web/
COPY api.py .

# Expose target port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch uvicorn on $PORT provided by platform (e.g. Render, Railway)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
