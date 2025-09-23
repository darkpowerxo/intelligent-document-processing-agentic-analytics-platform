"""Unit tests for streaming event schemas.

Tests the event schema definitions, validation, and factory methods.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from ai_architect_demo.streaming.event_schemas import (
    BaseEvent, DocumentEvent, TaskEvent, AgentEvent, AnalyticsEvent,
    EventType, Priority, EventFactory
)


class TestEventSchemas:
    """Test event schema models."""
    
    def test_base_event_creation(self):
        """Test basic event creation with required fields."""
        event = BaseEvent(
            event_type=EventType.DOCUMENT_UPLOADED,
            source="test_service"
        )
        
        assert event.event_type == EventType.DOCUMENT_UPLOADED.value
        assert event.source == "test_service"
        assert event.priority == Priority.MEDIUM
        assert isinstance(event.timestamp, datetime)
        assert len(event.event_id) > 0  # UUID should be generated
        assert isinstance(event.metadata, dict)
    
    def test_base_event_with_metadata(self):
        """Test event creation with custom metadata."""
        metadata = {"user_id": "123", "session_id": "abc"}
        event = BaseEvent(
            event_type=EventType.TASK_CREATED,
            source="test_service",
            metadata=metadata
        )
        
        assert event.metadata == metadata
    
    def test_document_event_creation(self):
        """Test document event creation."""
        event = DocumentEvent(
            event_type=EventType.DOCUMENT_UPLOADED,
            source="test_service",
            document_id="doc_123",
            document_name="test.pdf",
            document_type="pdf",
            file_size=1024
        )
        
        assert event.document_id == "doc_123"
        assert event.document_name == "test.pdf"
        assert event.document_type == "pdf"
        assert event.file_size == 1024
    
    def test_task_event_creation(self):
        """Test task event creation."""
        event = TaskEvent(
            event_type=EventType.TASK_CREATED,
            source="test_service",
            task_id="task_456",
            task_type="analysis",
            agent_id="agent_789"
        )
        
        assert event.task_id == "task_456"
        assert event.task_type == "analysis"
        assert event.agent_id == "agent_789"
    
    def test_agent_event_creation(self):
        """Test agent event creation."""
        event = AgentEvent(
            event_type=EventType.AGENT_STATUS_CHANGED,
            source="test_service",
            agent_id="agent_123",
            agent_name="Test Agent",
            agent_role="analyzer",
            status="working"
        )
        
        assert event.agent_id == "agent_123"
        assert event.agent_name == "Test Agent"
        assert event.agent_role == "analyzer"
        assert event.status == "working"
    
    def test_analytics_event_creation(self):
        """Test analytics event creation."""
        event = AnalyticsEvent(
            event_type=EventType.ANALYTICS_METRIC_UPDATE,
            source="test_service",
            metric_name="processing_time",
            metric_value=125.5,
            metric_type="gauge",
            service="document_processor"
        )
        
        assert event.metric_name == "processing_time"
        assert event.metric_value == 125.5
        assert event.metric_type == "gauge"
        assert event.service == "document_processor"


class TestEventFactory:
    """Test event factory methods."""
    
    def test_create_document_event(self):
        """Test document event factory method."""
        event = EventFactory.create_document_event(
            event_type=EventType.DOCUMENT_PROCESSED,
            document_id="doc_123",
            source="test_factory",
            document_name="test.pdf",
            file_size=2048
        )
        
        assert isinstance(event, DocumentEvent)
        assert event.event_type == EventType.DOCUMENT_PROCESSED.value
        assert event.document_id == "doc_123"
        assert event.source == "test_factory"
        assert event.document_name == "test.pdf"
        assert event.file_size == 2048
    
    def test_create_task_event(self):
        """Test task event factory method."""
        event = EventFactory.create_task_event(
            event_type=EventType.TASK_STARTED,
            task_id="task_456",
            task_type="analysis",
            source="test_factory"
        )
        
        assert isinstance(event, TaskEvent)
        assert event.event_type == EventType.TASK_STARTED.value
        assert event.task_id == "task_456"
        assert event.task_type == "analysis"
        assert event.source == "test_factory"
    
    def test_create_agent_event(self):
        """Test agent event factory method."""
        event = EventFactory.create_agent_event(
            event_type=EventType.AGENT_STATUS_CHANGED,
            agent_id="agent_789",
            agent_name="Test Agent",
            agent_role="processor",
            source="test_factory",
            status="idle"
        )
        
        assert isinstance(event, AgentEvent)
        assert event.event_type == EventType.AGENT_STATUS_CHANGED.value
        assert event.agent_id == "agent_789"
        assert event.agent_name == "Test Agent"
        assert event.status == "idle"
    
    def test_create_analytics_event(self):
        """Test analytics event factory method."""
        event = EventFactory.create_analytics_event(
            metric_name="error_rate",
            metric_value=0.05,
            metric_type="gauge",
            service="api_server",
            source="test_factory"
        )
        
        assert isinstance(event, AnalyticsEvent)
        assert event.event_type == EventType.ANALYTICS_METRIC_UPDATE.value
        assert event.metric_name == "error_rate"
        assert event.metric_value == 0.05
        assert event.service == "api_server"
    
    def test_factory_with_metadata(self):
        """Test factory methods with custom metadata."""
        metadata = {"correlation_id": "xyz", "user_id": "user123"}
        
        event = EventFactory.create_document_event(
            event_type=EventType.DOCUMENT_UPLOADED,
            document_id="doc_with_meta",
            source="test_factory",
            metadata=metadata
        )
        
        assert event.metadata == metadata


class TestEventValidation:
    """Test event validation and constraints."""
    
    def test_required_fields_validation(self):
        """Test that required fields are validated."""
        with pytest.raises(Exception):  # Pydantic validation error
            DocumentEvent(
                event_type=EventType.DOCUMENT_UPLOADED,
                source="test"
                # Missing required document_id
            )
    
    def test_enum_validation(self):
        """Test that enum fields are properly validated."""
        # Valid enum value should work
        event = BaseEvent(
            event_type=EventType.DOCUMENT_UPLOADED,
            source="test",
            priority=Priority.HIGH
        )
        assert event.priority == Priority.HIGH.value
    
    def test_serialization(self):
        """Test event serialization to dict."""
        event = DocumentEvent(
            event_type=EventType.DOCUMENT_PROCESSED,
            source="test_service",
            document_id="doc_123",
            document_name="test.pdf"
        )
        
        event_dict = event.model_dump()
        
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == "document.processed"
        assert event_dict["source"] == "test_service"
        assert event_dict["document_id"] == "doc_123"
        assert "event_id" in event_dict
        assert "timestamp" in event_dict
    
    def test_deserialization(self):
        """Test event deserialization from dict."""
        event_data = {
            "event_type": "document.uploaded",
            "source": "test_service",
            "document_id": "doc_123",
            "document_name": "test.pdf",
            "metadata": {"test": True}
        }
        
        event = DocumentEvent(**event_data)
        
        assert event.event_type == "document.uploaded"
        assert event.document_id == "doc_123"
        assert event.metadata["test"] is True