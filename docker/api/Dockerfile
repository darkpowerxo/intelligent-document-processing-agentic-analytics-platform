# FastAPI Application Dockerfile
# Multi-stage build for optimization

# Build stage - install dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies (combine into single layer)
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv package manager with specific version for reproducibility
RUN pip install --no-cache-dir uv==0.1.45

# Copy only dependency files first (for better caching)
COPY pyproject.toml ./
COPY README.md ./

# Copy source code needed for editable install
COPY ai_architect_demo ./ai_architect_demo/

# Create virtual environment and install dependencies
# This layer will be cached unless pyproject.toml changes
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/opt/venv uv pip install -e .

# Production stage - minimal runtime image
FROM python:3.11-slim AS production

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files (this happens last to maximize cache hits)
COPY pyproject.toml ./
COPY README.md ./
COPY ai_architect_demo ./ai_architect_demo/

# Create uploads directory
RUN mkdir -p /app/uploads

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "ai_architect_demo.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]