# ============================================================================
# RA-RL-IDS Dockerfile
# Multi-stage build for Resource-Aware RL IDS for IoMT Networks
# ============================================================================
# Usage:
#   docker build -t ra-rl-ids .
#   docker run --cpus=0.5 --memory=256m ra-rl-ids python src/train.py --reward_mode flat
# ============================================================================

FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# --- Dependencies stage ---
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application stage ---
FROM deps AS app

# Copy project source
COPY config.yaml .
COPY src/ src/
COPY tests/ tests/

# Create output directories
RUN mkdir -p data/raw data/processed results/figures checkpoints

# Default command: show help
CMD ["python", "-c", "print('RA-RL-IDS ready. Run: python src/train.py --reward_mode flat|weighted')"]
