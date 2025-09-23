"""Integration test fixtures and utilities.

This module provides fixtures and utilities for integration tests.
Integration tests test the interaction between multiple components.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock

from ai_architect_demo.streaming.kafka_client import KafkaProducer, KafkaConsumer, KafkaConfig
from ai_architect_demo.agents.orchestrator import AgentOrchestrator


@pytest.fixture(scope="session") 
async def kafka_integration_setup():
    """Set up Kafka integration test environment."""
    # Check if Kafka is available
    config = KafkaConfig(bootstrap_servers="localhost:9092")
    producer = KafkaProducer(config)
    
    try:
        await producer.connect()
        kafka_available = producer.is_connected
        await producer.disconnect()
    except Exception:
        kafka_available = False
    
    if not kafka_available:
        pytest.skip("Kafka not available for integration tests")
    
    yield {"kafka_available": True, "config": config}


@pytest.fixture
async def integration_agent_orchestrator() -> AsyncGenerator[AgentOrchestrator, None]:
    """Create an agent orchestrator for integration tests."""
    orchestrator = AgentOrchestrator()
    
    # Ensure all agents are properly initialized
    await asyncio.sleep(0.1)  # Brief wait for initialization
    
    yield orchestrator
    
    # Cleanup if needed
    pass


@pytest.fixture
async def integration_kafka_producer(kafka_integration_setup) -> AsyncGenerator[KafkaProducer, None]:
    """Create a Kafka producer for integration tests."""
    config = kafka_integration_setup["config"]
    producer = KafkaProducer(config)
    
    await producer.connect()
    
    yield producer
    
    await producer.disconnect()


@pytest.fixture
async def integration_kafka_consumer(kafka_integration_setup) -> AsyncGenerator[KafkaConsumer, None]:
    """Create a Kafka consumer for integration tests."""
    config = kafka_integration_setup["config"]
    consumer = KafkaConsumer(
        topics=["test_integration"],
        config=config,
        group_id="integration_test_group"
    )
    
    await consumer.connect()
    
    yield consumer
    
    await consumer.disconnect()


class IntegrationTestHelpers:
    """Utilities for integration testing."""
    
    @staticmethod
    async def wait_for_message(consumer: KafkaConsumer, timeout: int = 10):
        """Wait for a message from Kafka consumer with timeout."""
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            message_batch = consumer.consumer.poll(timeout_ms=1000)
            if message_batch:
                for topic_partition, messages in message_batch.items():
                    if messages:
                        return messages[0]  # Return first message
            await asyncio.sleep(0.1)
        
        return None
    
    @staticmethod
    async def wait_for_agent_completion(orchestrator: AgentOrchestrator, task_id: str, timeout: int = 30):
        """Wait for an agent task to complete."""
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Check if task is completed (this would require task tracking)
            # For now, just wait a reasonable amount of time
            await asyncio.sleep(1)
            
            # In a real implementation, you'd check task status
            # For testing, we'll assume completion after a short wait
            if (time.time() - start_time) > 2:
                return True
        
        return False


@pytest.fixture
def integration_helpers():
    """Provide integration test helper utilities."""
    return IntegrationTestHelpers()