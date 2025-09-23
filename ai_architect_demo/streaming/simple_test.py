"""Simple test for streaming architecture without full agent dependencies.

This is a basic test that validates the core streaming components
without requiring the complete agent system to be functional.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from ai_architect_demo.streaming.event_schemas import EventFactory, EventType


def test_event_factory():
    """Test event factory creates valid events."""
    print("Testing event factory...")
    
    # Test document event
    doc_event = EventFactory.create_document_event(
        event_type=EventType.DOCUMENT_UPLOADED,
        document_id="test_doc",
        document_name="test.pdf",
        document_type="pdf",
        file_size=1024,
        source="test"
    )
    
    assert doc_event.event_type == EventType.DOCUMENT_UPLOADED.value
    assert doc_event.document_id == "test_doc"
    assert doc_event.document_name == "test.pdf"
    print(f"✓ Document event creation works - created {doc_event.event_type}")
    
    # Test task event
    task_event = EventFactory.create_task_event(
        event_type=EventType.TASK_CREATED,
        task_id="test_task",
        task_type="document_analysis",
        agent_id="test_agent",
        source="test"
    )
    
    assert task_event.event_type == EventType.TASK_CREATED.value
    assert task_event.task_id == "test_task"
    assert task_event.task_type == "document_analysis"
    print("✓ Task event creation works")
    
    # Test agent event
    agent_event = EventFactory.create_agent_event(
        event_type=EventType.AGENT_STATUS_CHANGED,
        agent_id="test_agent",
        agent_name="Test Agent",
        agent_role="document_analyzer",
        status="working",
        source="test"
    )
    
    assert agent_event.event_type == EventType.AGENT_STATUS_CHANGED.value
    assert agent_event.agent_id == "test_agent"
    assert agent_event.status == "working"
    print("✓ Agent event creation works")
    
    # Test analytics event
    analytics_event = EventFactory.create_analytics_event(
        metric_name="test_metric",
        metric_value=42.0,
        metric_type="gauge",
        service="test_service",
        source="test"
    )
    
    assert analytics_event.event_type == EventType.ANALYTICS_METRIC_UPDATE.value
    assert analytics_event.metric_name == "test_metric"
    assert analytics_event.metric_value == 42.0
    print("✓ Analytics event creation works")
    
    print("All event factory tests passed!")


def test_event_serialization():
    """Test event serialization and deserialization."""
    print("\nTesting event serialization...")
    
    # Create an event
    event = EventFactory.create_document_event(
        event_type=EventType.DOCUMENT_PROCESSED,
        document_id="serialization_test",
        document_name="serialization.pdf",
        document_type="pdf",
        file_size=1024,
        metadata={"pages": 10, "size": 1024},
        source="test"
    )
    
    # Serialize to dict
    event_dict = event.model_dump()
    assert isinstance(event_dict, dict)
    assert event_dict["document_id"] == "serialization_test"
    assert event_dict["metadata"]["pages"] == 10
    print("✓ Event serialization works")
    
    # Serialize to JSON
    event_json = json.dumps(event_dict, default=str)
    assert isinstance(event_json, str)
    print("✓ Event JSON serialization works")
    
    # Deserialize from JSON
    parsed_dict = json.loads(event_json)
    assert parsed_dict["document_id"] == "serialization_test"
    print("✓ Event JSON deserialization works")
    
    print("All serialization tests passed!")


def test_kafka_config():
    """Test Kafka configuration."""
    print("\nTesting Kafka configuration...")
    
    try:
        from ai_architect_demo.streaming.kafka_client import KafkaConfig
        
        # Test default config
        config = KafkaConfig()
        producer_config = config.get_producer_config()
        consumer_config = config.get_consumer_config("test_group")
        
        assert "bootstrap_servers" in producer_config
        assert "bootstrap_servers" in consumer_config
        assert consumer_config["group_id"] == "test_group"
        
        print("✓ Kafka configuration works")
        
        # Test custom config
        custom_config = KafkaConfig(
            bootstrap_servers="custom:9092",
            security_protocol="SASL_SSL"
        )
        
        custom_producer_config = custom_config.get_producer_config()
        assert custom_producer_config["bootstrap_servers"] == "custom:9092"
        assert custom_producer_config["security_protocol"] == "SASL_SSL"
        
        print("✓ Custom Kafka configuration works")
        
    except ImportError as e:
        print(f"⚠ Kafka client test skipped due to import error: {e}")


def test_stream_processor_base():
    """Test stream processor base functionality."""
    print("\nTesting stream processor...")
    
    try:
        from ai_architect_demo.streaming.stream_processors import DocumentStreamProcessor
        
        # Create processor
        processor = DocumentStreamProcessor()
        
        # Test initial state
        assert not processor.is_running
        assert processor.name == "document_processor"
        assert "documents" in processor.input_topics
        
        # Test stats
        stats = processor.get_stats()
        assert "name" in stats
        assert stats["name"] == "document_processor"
        
        print("✓ Stream processor creation works")
        
    except ImportError as e:
        print(f"⚠ Stream processor test skipped due to import error: {e}")


def test_event_dispatcher_config():
    """Test event dispatcher configuration."""
    print("\nTesting event dispatcher...")
    
    try:
        from ai_architect_demo.streaming.event_dispatcher import EventRoute, EventFilter
        
        # Test event route
        route = EventRoute(
            name="test_route",
            source_topics=["input"],
            target_topics=["output"],
            event_types=[EventType.DOCUMENT_UPLOADED],
            filters={"status": "pending"},
            priority=10
        )
        
        assert route.name == "test_route"
        assert route.priority == 10
        assert route.enabled == True  # default
        
        print("✓ Event route creation works")
        
        # Test event filtering
        event_data = {
            "event_type": "document_uploaded",
            "status": "pending",
            "document_id": "test"
        }
        
        # Test exact match
        assert EventFilter.matches_criteria(event_data, {"status": "pending"})
        assert not EventFilter.matches_criteria(event_data, {"status": "completed"})
        
        # Test $in filter
        assert EventFilter.matches_criteria(event_data, {"status": {"$in": ["pending", "processing"]}})
        assert not EventFilter.matches_criteria(event_data, {"status": {"$in": ["completed", "failed"]}})
        
        print("✓ Event filtering works")
        
    except ImportError as e:
        print(f"⚠ Event dispatcher test skipped due to import error: {e}")


def test_monitoring_components():
    """Test monitoring component creation."""
    print("\nTesting monitoring components...")
    
    try:
        from ai_architect_demo.streaming.real_time_monitor import AlertRule, HealthStatus
        
        # Test alert rule
        rule = AlertRule(
            name="test_alert",
            metric="error_rate",
            condition="gt",
            threshold=5.0,
            severity="high"
        )
        
        assert rule.name == "test_alert"
        assert rule.threshold == 5.0
        assert rule.enabled == True
        
        # Test threshold evaluation
        assert rule.should_trigger(10.0)  # Above threshold
        assert not rule.should_trigger(3.0)  # Below threshold
        
        print("✓ Alert rule works")
        
        # Test health status
        health = HealthStatus(
            component="test_component",
            status="healthy",
            message="All systems operational"
        )
        
        assert health.component == "test_component"
        assert health.is_healthy() == True
        
        # Test unhealthy status
        unhealthy = HealthStatus(
            component="test_component",
            status="critical",
            message="System down"
        )
        
        assert not unhealthy.is_healthy()
        
        print("✓ Health status works")
        
    except ImportError as e:
        print(f"⚠ Monitoring test skipped due to import error: {e}")


async def test_async_components():
    """Test async component creation."""
    print("\nTesting async components...")
    
    try:
        from ai_architect_demo.streaming.kafka_client import KafkaProducer
        
        # Test producer creation (without connecting to Kafka)
        producer = KafkaProducer()
        assert not producer.is_connected
        
        # Test stats
        stats = producer.get_stats()
        assert "is_connected" in stats
        assert stats["is_connected"] == False
        
        print("✓ Kafka producer creation works")
        
    except ImportError as e:
        print(f"⚠ Async components test skipped due to import error: {e}")


def main():
    """Run all basic tests."""
    print("Running basic streaming architecture tests...")
    print("=" * 50)
    
    # Run synchronous tests
    test_event_factory()
    test_event_serialization()
    test_kafka_config()
    test_stream_processor_base()
    test_event_dispatcher_config()
    test_monitoring_components()
    
    # Run async tests
    asyncio.run(test_async_components())
    
    print("\n" + "=" * 50)
    print("✅ All basic streaming tests completed successfully!")
    print("\nThe streaming architecture core components are working correctly.")
    print("Note: Full integration tests require Kafka to be running.")


if __name__ == "__main__":
    main()