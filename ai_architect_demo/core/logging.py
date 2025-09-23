"""Structured logging configuration for AI Architect Demo.

This module provides enterprise-grade logging with structured output,
correlation IDs, and integration with monitoring systems.
"""

import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from ai_architect_demo.core.config import settings

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer() if not settings.debug 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for request tracing.
    
    Args:
        corr_id: Correlation ID to set
    """
    correlation_id.set(corr_id)


def get_correlation_id() -> str:
    """Get current correlation ID.
    
    Returns:
        Current correlation ID or empty string if not set
    """
    return correlation_id.get()


class LoggerMixin:
    """Mixin class to add structured logging to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger bound to this class."""
        return get_logger(self.__class__.__module__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger()
    logger.info(
        "Function called",
        function=func_name,
        parameters=kwargs,
        correlation_id=get_correlation_id(),
    )


def log_performance(operation: str, duration: float, **metadata: Any) -> None:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **metadata: Additional metadata to log
    """
    logger = get_logger()
    logger.info(
        "Performance metric",
        operation=operation,
        duration_seconds=duration,
        correlation_id=get_correlation_id(),
        **metadata,
    )


def log_business_event(event_type: str, **data: Any) -> None:
    """Log business events for analytics.
    
    Args:
        event_type: Type of business event
        **data: Event data
    """
    logger = get_logger()
    logger.info(
        "Business event",
        event_type=event_type,
        correlation_id=get_correlation_id(),
        **data,
    )


# Initialize logging configuration
configure_logging()