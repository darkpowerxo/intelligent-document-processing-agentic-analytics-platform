"""API server startup script.

This script starts the FastAPI server with proper configuration
for the AI Architect Demo application.
"""

import uvicorn
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger

logger = get_logger(__name__)


def main():
    """Start the FastAPI server."""
    logger.info("Starting AI Architect Demo API server...")
    
    # Configure uvicorn
    config = {
        "app": "ai_architect_demo.api.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
        "reload_dirs": [str(project_root / "ai_architect_demo")],
    }
    
    # Start server
    uvicorn.run(**config)


if __name__ == "__main__":
    main()