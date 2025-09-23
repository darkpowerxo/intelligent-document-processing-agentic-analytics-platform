"""Simple unit tests for streaming event schemas."""

import pytest
from datetime import datetime
from uuid import UUID

from ai_architect_demo.streaming.event_schemas import (
    BaseEvent, 
    DocumentEvent, 
    TaskEvent,
    EventType,
    EventFactory
)


class TestBaseEvent:
    """Test BaseEvent functionality."""
    
    def test_base_event_creation(self):
        """Test creating a base event."""
        event = BaseEvent(
            event_type=EventType.DOCUMENT_UPLOADED,
            source="test_service",
            metadata={"test": "data"}
        )
        
        assert event.event_type == EventType.DOCUMENT_UPLOADED.value  # Compare with enum value
        assert event.source == "test_service"
        assert event.metadata == {"test": "data"}
        assert isinstance(event.event_id, str)
        assert isinstance(event.timestamp, datetime)
    
    def test_event_serialization(self):
        """Test event serialization to dict."""
        event = BaseEvent(
            event_type=EventType.DOCUMENT_UPLOADED,
            source="test_service",
            metadata={"test": "data"}
        )
        
        event_dict = event.model_dump()
        
        assert "event_id" in event_dict
        assert event_dict["event_type"] == "document.uploaded"
        assert event_dict["source"] == "test_service"
        assert event_dict["metadata"] == {"test": "data"}


class TestDocumentEvent:
    """Test DocumentEvent functionality."""
    
    def test_document_event_creation(self):
        """Test creating a document event."""
        event = DocumentEvent(
            document_id="doc123",
            document_name="test.pdf", 
            event_type=EventType.DOCUMENT_UPLOADED,
            source="document_service"
        )
        
        assert event.document_id == "doc123"
        assert event.document_name == "test.pdf"
        assert event.event_type == EventType.DOCUMENT_UPLOADED.value
        assert isinstance(event.event_id, str)
    
    def test_document_event_with_optional_fields(self):
        """Test document event with optional fields."""
        event = DocumentEvent(
            document_id="doc123",
            document_name="test.pdf",
            event_type=EventType.DOCUMENT_UPLOADED,
            source="document_service",
            file_size=1024,
            user_id="user123"
        )
        
        assert event.file_size == 1024
        assert event.user_id == "user123"


class TestTaskEvent:
    """Test TaskEvent functionality."""
    
    def test_task_event_creation(self):
        """Test creating a task event."""
        event = TaskEvent(
            task_id="task123",
            task_type="document_analysis",
            event_type=EventType.TASK_CREATED,
            source="task_service"
        )
        
        assert event.task_id == "task123"
        assert event.task_type == "document_analysis"
        assert event.event_type == EventType.TASK_CREATED.value


class TestEventFactory:
    """Test EventFactory functionality."""
    
    def test_create_document_event(self):
        """Test factory method for document event."""
        event = EventFactory.create_document_event(
            document_id="doc123",
            event_type=EventType.DOCUMENT_UPLOADED,
            source="upload_service",
            document_name="test.pdf"
        )
        
        assert isinstance(event, DocumentEvent)
        assert event.document_id == "doc123"
        assert event.document_name == "test.pdf"
        assert event.event_type == EventType.DOCUMENT_UPLOADED.value
    
    def test_create_task_event(self):
        """Test factory method for task event."""
        event = EventFactory.create_task_event(
            task_id="task123",
            task_type="analysis",
            event_type=EventType.TASK_CREATED,
            source="task_service"
        )
        
        assert isinstance(event, TaskEvent)
        assert event.task_id == "task123"
        assert event.task_type == "analysis"
        assert event.event_type == EventType.TASK_CREATED.value