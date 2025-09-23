"""Integration tests for streaming and agents.

Tests the integration between streaming components and agent system.
"""

import pytest
import asyncio
import json
from typing import Dict, Any

from ai_architect_demo.streaming.event_schemas import EventFactory, EventType
from ai_architect_demo.agents.base import AgentTask, TaskPriority
from ai_architect_demo.agents.orchestrator import AgentOrchestrator
from ai_architect_demo.streaming.kafka_client import KafkaProducer, KafkaConsumer


@pytest.mark.integration
@pytest.mark.kafka
class TestStreamingAgentIntegration:
    """Test integration between streaming system and agents."""
    
    @pytest.mark.asyncio
    async def test_event_to_agent_task_flow(self, 
                                           integration_kafka_producer: KafkaProducer,
                                           integration_kafka_consumer: KafkaConsumer,
                                           integration_agent_orchestrator: AgentOrchestrator,
                                           integration_helpers):
        """Test complete flow from streaming event to agent task completion."""
        
        # 1. Create a document event
        document_event = EventFactory.create_document_event(
            event_type=EventType.DOCUMENT_UPLOADED,
            document_id="integration_doc_123",
            document_name="integration_test.pdf",
            document_type="pdf",
            source="integration_test",
            metadata={"test_type": "integration"}
        )
        
        # 2. Send event to Kafka
        await integration_kafka_producer.send_event(
            topic="test_integration",
            event=document_event
        )
        
        # 3. Consume the event
        message = await integration_helpers.wait_for_message(
            integration_kafka_consumer, 
            timeout=15
        )
        
        assert message is not None, "Failed to receive message from Kafka"
        
        # 4. Parse event and create agent task
        event_data = json.loads(message.value.decode('utf-8'))
        assert event_data["document_id"] == "integration_doc_123"
        
        # 5. Create agent task based on event
        agent_task = AgentTask(
            task_id=f"task_{event_data['document_id']}",
            task_type="document_analysis",
            priority=TaskPriority.MEDIUM,
            data={
                "document_id": event_data["document_id"],
                "document_name": event_data["document_name"],
                "content": "Sample content for integration testing"
            },
            requester_id="integration_test"
        )
        
        # 6. Submit task to orchestrator
        result = await integration_agent_orchestrator.submit_task(agent_task)
        
        # 7. Verify task completion
        assert result["status"] == "completed"
        assert "analysis" in result or "result" in result
    
    @pytest.mark.asyncio
    async def test_multiple_event_types_processing(self,
                                                  integration_kafka_producer: KafkaProducer,
                                                  integration_kafka_consumer: KafkaConsumer,
                                                  integration_agent_orchestrator: AgentOrchestrator,
                                                  integration_helpers):
        """Test processing of multiple different event types."""
        
        events = []
        
        # Create different types of events
        doc_event = EventFactory.create_document_event(
            event_type=EventType.DOCUMENT_PROCESSED,
            document_id="multi_doc_456",
            document_name="multi_test.pdf",
            document_type="pdf",
            source="multi_test"
        )
        events.append(("document", doc_event))
        
        task_event = EventFactory.create_task_event(
            event_type=EventType.TASK_CREATED,
            task_id="multi_task_789",
            task_type="business_analysis",
            source="multi_test"
        )
        events.append(("task", task_event))
        
        # Send all events
        for event_type, event in events:
            await integration_kafka_producer.send_event(
                topic="test_integration",
                event=event
            )
        
        # Process events as they arrive
        processed_events = []
        for _ in range(len(events)):
            message = await integration_helpers.wait_for_message(
                integration_kafka_consumer,
                timeout=10
            )
            
            if message:
                event_data = json.loads(message.value.decode('utf-8'))
                processed_events.append(event_data)
        
        # Verify all events were processed
        assert len(processed_events) >= len(events)
        
        # Verify event types are correct
        event_types = [e["event_type"] for e in processed_events]
        assert "document.processed" in event_types
        assert "task.created" in event_types


@pytest.mark.integration
class TestAgentOrchestrationIntegration:
    """Test agent orchestration integration."""
    
    @pytest.mark.asyncio
    async def test_agent_task_routing(self, integration_agent_orchestrator: AgentOrchestrator):
        """Test that tasks are correctly routed to appropriate agents."""
        
        # Test document analysis routing
        doc_task = AgentTask(
            task_id="routing_doc_task",
            task_type="document_analysis",
            priority=TaskPriority.HIGH,
            data={
                "document_id": "routing_test_doc",
                "content": "This is content for routing test"
            },
            requester_id="routing_test"
        )
        
        result = await integration_agent_orchestrator.submit_task(doc_task)
        assert result["status"] == "completed"
        assert result["agent"] == "document_analyzer"
        
        # Test business intelligence routing  
        bi_task = AgentTask(
            task_id="routing_bi_task",
            task_type="business_analysis", 
            priority=TaskPriority.MEDIUM,
            data={
                "data_source": "sales_data",
                "metrics": ["revenue", "growth"]
            },
            requester_id="routing_test"
        )
        
        result = await integration_agent_orchestrator.submit_task(bi_task)
        assert result["status"] == "completed"
        assert result["agent"] == "business_intelligence"
        
        # Test quality assurance routing
        qa_task = AgentTask(
            task_id="routing_qa_task",
            task_type="quality_check",
            priority=TaskPriority.HIGH,
            data={
                "subject_type": "analysis",
                "subject_id": "test_analysis",
                "validation_rules": ["completeness", "accuracy"]
            },
            requester_id="routing_test"
        )
        
        result = await integration_agent_orchestrator.submit_task(qa_task)
        assert result["status"] == "completed"
        assert result["agent"] == "quality_assurance"
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, integration_agent_orchestrator: AgentOrchestrator):
        """Test processing multiple tasks concurrently."""
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = AgentTask(
                task_id=f"concurrent_task_{i}",
                task_type="document_analysis",
                priority=TaskPriority.MEDIUM,
                data={
                    "document_id": f"concurrent_doc_{i}",
                    "content": f"Content for concurrent test {i}"
                },
                requester_id="concurrent_test"
            )
            tasks.append(task)
        
        # Submit all tasks concurrently
        task_coroutines = [
            integration_agent_orchestrator.submit_task(task) 
            for task in tasks
        ]
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Verify all tasks completed successfully
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_agent_availability_tracking(self, integration_agent_orchestrator: AgentOrchestrator):
        """Test that agent availability is properly tracked."""
        
        # Get initial availability
        available_agents = integration_agent_orchestrator.get_available_agents()
        initial_count = len(available_agents)
        
        assert initial_count == 3  # doc, bi, qa agents
        
        # Verify all agents report as available
        for agent_id, agent in available_agents.items():
            assert agent.is_available is True
        
        # Submit a task and check capabilities
        capabilities = integration_agent_orchestrator.get_agent_capabilities()
        
        assert "document_analyzer" in capabilities
        assert "business_intelligence" in capabilities  
        assert "quality_assurance" in capabilities
        
        # Each agent should have specific capabilities
        doc_caps = capabilities["document_analyzer"]
        assert "document_analysis" in doc_caps
        
        bi_caps = capabilities["business_intelligence"]
        assert "business_analysis" in bi_caps
        
        qa_caps = capabilities["quality_assurance"] 
        assert "quality_check" in qa_caps


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Test database integration with components."""
    
    @pytest.mark.asyncio
    async def test_task_persistence(self, db_session):
        """Test that tasks can be persisted and retrieved."""
        # This would test actual database operations
        # For now, we'll test that the session is available
        assert db_session is not None
        
        # In a real implementation, you would:
        # 1. Create a task record
        # 2. Save it to database  
        # 3. Retrieve and verify
        pass
    
    @pytest.mark.asyncio  
    async def test_agent_metrics_storage(self, db_session):
        """Test that agent metrics are properly stored."""
        # This would test storing agent performance metrics
        assert db_session is not None
        pass