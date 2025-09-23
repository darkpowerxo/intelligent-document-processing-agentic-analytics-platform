"""Unit test fixtures and utilities."""

import pytest
from unittest.mock import Mock, AsyncMock


# Mock fixtures for unit testing
@pytest.fixture  
def mock_agent():
    """Mock agent for testing."""
    agent = Mock()
    agent.process_task = AsyncMock(return_value={"status": "completed"})
    agent.get_status = Mock(return_value="idle")
    agent.agent_id = "test_agent"
    return agent


@pytest.fixture
def mock_task():
    """Mock task for testing."""
    return {
        "task_id": "test_task_123",
        "task_type": "document_analysis", 
        "priority": "high",
        "data": {"document_id": "doc123"},
        "created_at": "2024-01-01T00:00:00Z"
    }