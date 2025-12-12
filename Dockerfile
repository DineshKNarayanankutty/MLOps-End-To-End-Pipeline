# ---------- Stage 1: Builder ----------
FROM python:3.10-slim AS builder
LABEL maintainer="MLOps Team"
WORKDIR /build

# Build deps only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /build/

# Install packages into /build/install (no --user)
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    --prefix=/build/install \
    --compile \
    -r /build/requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.10-slim
LABEL maintainer="MLOps Team"
WORKDIR /app

# runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early so we can use --chown when copying app files
RUN useradd --create-home --shell /bin/bash mlops

# Copy installed packages from builder into /usr/local
# Then fix permissions so non-root user can execute installed CLIs
COPY --from=builder /build/install /usr/local
RUN chown -R root:root /usr/local && chmod -R a+rX /usr/local && chown -R mlops:mlops /usr/local/bin || true

# Copy app source (but avoid copying large local dirs - use .dockerignore)
COPY --chown=mlops:mlops . /app

# Ensure /app ownership and python can write to logs/models if needed
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

ENV PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
