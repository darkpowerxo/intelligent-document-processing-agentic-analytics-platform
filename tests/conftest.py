"""Test configuration and shared fixtures for AI Architecture tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock


@pytest.fixture
def test_settings():
    """Test configuration settings."""
    from ai_architect_demo.core.config import Settings
    return Settings(
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        environment="testing"
    )


@pytest.fixture
def sample_documents():
    """Create sample test documents."""
    test_files = {}
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create sample text file
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is a sample text document for testing purposes.")
    test_files["text"] = text_file
    
    # Create sample markdown file
    md_file = temp_dir / "sample.md"
    md_file.write_text("# Sample Document\n\nThis is a **sample** markdown document.")
    test_files["markdown"] = md_file
    
    return test_files


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "documents": [
            {"id": "doc1", "title": "Test Document 1", "content": "Content 1"},
            {"id": "doc2", "title": "Test Document 2", "content": "Content 2"}
        ]
    }