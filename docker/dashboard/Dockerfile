# Streamlit Dashboard Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY ai_architect_demo ./ai_architect_demo/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e .

# Create non-root user and home directory
RUN groupadd -r appuser && useradd -r -g appuser -m appuser
RUN chown -R appuser:appuser /app
RUN mkdir -p /home/appuser/.streamlit && chown -R appuser:appuser /home/appuser
USER appuser

# Expose port
EXPOSE 8501

# Start the Streamlit dashboard
CMD ["streamlit", "run", "ai_architect_demo/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]