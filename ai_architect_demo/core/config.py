"""Core configuration management for AI Architect Demo.

This module provides centralized configuration management using Pydantic settings
with support for environment variables, validation, and type safety.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "AI Architect Demo"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Database Settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ai_demo"
    postgres_password: str = "ai_demo_password"
    postgres_db: str = "ai_demo"
    
    # Redis Settings  
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # MLflow Settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "./mlruns"
    mlflow_experiment_name: str = "document_processing"
    
    # Ollama LLM Settings
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "llama3.1:latest"
    ollama_timeout: int = 120
    
    # Kafka Settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "ai_demo_group"
    # Topics as individual fields to avoid dict parsing issues
    kafka_topic_document_upload: str = "document-upload"
    kafka_topic_processing_results: str = "processing-results"
    kafka_topic_agent_comms: str = "agent-comms"
    
    # MinIO Settings
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "ai-demo-documents"
    minio_secure: bool = False
    
    # File Processing Settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: str = ".pdf,.docx,.txt,.md,.csv"  # Comma-separated string
    upload_dir: Path = Path("./uploads")
    
    # Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_sequence_length: int = 512
    batch_size: int = 32
    
    # Agent Settings
    max_agent_retries: int = 3
    agent_timeout: int = 60
    max_concurrent_agents: int = 5
    
    # Monitoring Settings
    prometheus_port: int = 8001
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Security Settings
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    @field_validator("upload_dir")
    @classmethod
    def create_upload_dir(cls, v):
        """Create upload directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def ollama_base_url(self) -> str:
        """Construct Ollama base URL."""
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    @property
    def kafka_servers(self) -> List[str]:
        """Get Kafka bootstrap servers as a list."""
        return [server.strip() for server in self.kafka_bootstrap_servers.split(",")]
    
    @property
    def file_types_list(self) -> List[str]:
        """Get allowed file types as a list."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]
    
    @property
    def kafka_topics_dict(self) -> dict:
        """Get Kafka topics as a dictionary."""
        return {
            "document_upload": self.kafka_topic_document_upload,
            "processing_results": self.kafka_topic_processing_results,
            "agent_communications": self.kafka_topic_agent_comms
        }
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings