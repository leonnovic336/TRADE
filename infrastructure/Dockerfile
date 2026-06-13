# OMNI-TRADE AI - Dockerfile
# Multi-stage build for optimal image size

# ==========================================
# Stage 1: Builder
# ==========================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    build-essential cmake git curl wget \
    libffi-dev libssl-dev \
    librdkafka-dev rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download AI models (will be cached)
RUN python3.12 -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')" || true
RUN python3.12 -c "import whisper; whisper.load_model('base')" || true

# ==========================================
# Stage 2: Production
# ==========================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 libgomp1 libffi7 libssl3 \
    librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src /app/src
COPY config /app/config
COPY models /app/models 2>/dev/null || true

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader
RUN mkdir -p /app/logs /app/data /var/log/audit && \
    chown -R trader:trader /app /var/log/audit

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.12 -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Expose ports
EXPOSE 8000 9090 10000

# Default command
CMD ["python3.12", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Stage 3: Development (optional)
# ==========================================
FROM production AS development

USER root
RUN apt-get update && apt-get install -y \
    vim less strace tcpdump \
    && rm -rf /var/lib/apt/lists/*
USER trader

# Development tools
RUN pip install --no-cache-dir ipython jupyter

CMD ["python3.12", "-m", "ipython"]