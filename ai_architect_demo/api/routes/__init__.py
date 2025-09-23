"""API routes package for AI Architect Demo.

This package provides organized API route modules for:
- Document management and processing
- Model management and predictions
- Analytics and monitoring
- User management and authentication
"""

from .documents import router as documents_router
from .models import router as models_router

__all__ = ["documents_router", "models_router"]