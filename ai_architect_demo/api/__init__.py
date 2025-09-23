"""API package for AI Architect Demo.

This package provides the FastAPI-based REST API with:
- Document upload and processing endpoints
- Model serving and prediction endpoints
- Authentication and authorization
- Real-time processing status
- Comprehensive API documentation
"""

from .main import app

__all__ = ["app"]