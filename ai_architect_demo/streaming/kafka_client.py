"""Kafka client wrapper for streaming operations.

Provides high-level, production-ready Kafka producer and consumer
implementations with error handling, retry logic, and monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

import kafka
from kafka import KafkaProducer as KafkaProducerClient
from kafka import KafkaConsumer as KafkaConsumerClient
from kafka.errors import KafkaError, KafkaTimeoutError

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.streaming.event_schemas import BaseEvent, EventType

logger = get_logger(__name__)


class KafkaConfig:
    """Kafka configuration settings."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        security_protocol: str = "PLAINTEXT",
        **kwargs
    ):
        self.bootstrap_servers = bootstrap_servers
        self.security_protocol = security_protocol
        self.extra_config = kwargs
    
    def get_producer_config(self) -> Dict[str, Any]:
        """Get Kafka producer configuration."""
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'security_protocol': self.security_protocol,
            'value_serializer': lambda v: json.dumps(v, default=str).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'retry_backoff_ms': 100,
            'linger_ms': 10,  # Batch messages for efficiency
            'batch_size': 16384,
            'buffer_memory': 33554432,
            'max_in_flight_requests_per_connection': 1,  # Ensure ordering
            'enable_idempotence': True,  # Prevent duplicates
        }
        config.update(self.extra_config)
        return config
    
    def get_consumer_config(self, group_id: str) -> Dict[str, Any]:
        """Get Kafka consumer configuration."""
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'security_protocol': self.security_protocol,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'group_id': group_id,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,  # Manual commit for reliability
            'max_poll_records': 100,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 10000,
            'fetch_min_bytes': 1,
            'fetch_max_wait_ms': 500,
        }
        config.update(self.extra_config)
        return config


class KafkaProducer:
    """High-level Kafka producer for event streaming."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """Initialize Kafka producer.
        
        Args:
            config: Kafka configuration, defaults to local setup
        """
        self.config = config or KafkaConfig()
        self.producer: Optional[KafkaProducerClient] = None
        self.is_connected = False
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'connection_errors': 0
        }
    
    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        log_function_call("connect", producer_id=id(self))
        
        try:
            producer_config = self.config.get_producer_config()
            self.producer = KafkaProducerClient(**producer_config)
            
            # Test connection with a metadata request
            metadata = self.producer.bootstrap_connected()
            if metadata:
                self.is_connected = True
                logger.info("Kafka producer connected successfully")
            else:
                raise Exception("Failed to connect to Kafka bootstrap servers")
                
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"Failed to connect Kafka producer: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        log_function_call("disconnect", producer_id=id(self))
        
        if self.producer:
            try:
                self.producer.flush(timeout=10)  # Ensure all messages are sent
                self.producer.close(timeout=10)
                logger.info("Kafka producer disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Kafka producer: {e}")
            finally:
                self.producer = None
                self.is_connected = False
    
    async def send_event(
        self,
        topic: str,
        event: BaseEvent,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Send an event to a Kafka topic.
        
        Args:
            topic: Kafka topic name
            event: Event to send
            key: Optional message key for partitioning
            partition: Optional specific partition
            timestamp: Optional message timestamp
            
        Returns:
            True if message was sent successfully
        """
        if not self.is_connected or not self.producer:
            await self.connect()
        
        try:
            # Convert event to dict
            event_data = event.model_dump()
            
            # Add metadata
            event_data['_kafka_metadata'] = {
                'topic': topic,
                'key': key,
                'timestamp': (timestamp or datetime.now()).isoformat(),
                'producer_id': id(self)
            }
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=event_data,
                key=key,
                partition=partition,
                timestamp_ms=int((timestamp or datetime.now()).timestamp() * 1000)
            )
            
            # Wait for send to complete (with timeout)
            record_metadata = future.get(timeout=10)
            
            # Update statistics
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += len(json.dumps(event_data, default=str))
            
            logger.debug(
                f"Event sent to {topic} (partition {record_metadata.partition}, "
                f"offset {record_metadata.offset})"
            )
            
            return True
            
        except (KafkaTimeoutError, KafkaError) as e:
            self.stats['messages_failed'] += 1
            logger.error(f"Failed to send event to {topic}: {e}")
            return False
        except Exception as e:
            self.stats['messages_failed'] += 1
            logger.error(f"Unexpected error sending event to {topic}: {e}")
            return False
    
    async def send_batch(
        self,
        topic: str,
        events: List[BaseEvent],
        key_func: Optional[Callable[[BaseEvent], str]] = None
    ) -> Dict[str, int]:
        """Send a batch of events to a topic.
        
        Args:
            topic: Kafka topic name
            events: List of events to send
            key_func: Optional function to generate keys from events
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {'success': 0, 'failed': 0}
        
        for event in events:
            key = key_func(event) if key_func else None
            success = await self.send_event(topic, event, key=key)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Batch send to {topic}: {results['success']} success, {results['failed']} failed")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            'is_connected': self.is_connected,
            'stats': self.stats.copy(),
            'config': {
                'bootstrap_servers': self.config.bootstrap_servers,
                'security_protocol': self.config.security_protocol
            }
        }


class KafkaConsumer:
    """High-level Kafka consumer for event streaming."""
    
    def __init__(
        self,
        topics: Union[str, List[str]],
        group_id: str,
        config: Optional[KafkaConfig] = None
    ):
        """Initialize Kafka consumer.
        
        Args:
            topics: Topic(s) to subscribe to
            group_id: Consumer group ID
            config: Kafka configuration
        """
        self.topics = [topics] if isinstance(topics, str) else topics
        self.group_id = group_id
        self.config = config or KafkaConfig()
        self.consumer: Optional[KafkaConsumerClient] = None
        self.is_running = False
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.stats = {
            'messages_consumed': 0,
            'messages_processed': 0,
            'processing_errors': 0,
            'connection_errors': 0
        }
    
    async def connect(self) -> None:
        """Connect to Kafka cluster and subscribe to topics."""
        log_function_call("connect", consumer_group=self.group_id)
        
        try:
            consumer_config = self.config.get_consumer_config(self.group_id)
            self.consumer = KafkaConsumerClient(**consumer_config)
            
            # Subscribe to topics
            self.consumer.subscribe(self.topics)
            
            logger.info(f"Kafka consumer connected to topics: {self.topics}")
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"Failed to connect Kafka consumer: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        log_function_call("disconnect", consumer_group=self.group_id)
        
        self.is_running = False
        
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Kafka consumer disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Kafka consumer: {e}")
            finally:
                self.consumer = None
    
    def add_handler(self, event_type: Union[str, EventType], handler: Callable) -> None:
        """Add a message handler for specific event types.
        
        Args:
            event_type: Event type to handle
            handler: Async function to handle the event
        """
        event_key = event_type.value if isinstance(event_type, EventType) else event_type
        
        if event_key not in self.message_handlers:
            self.message_handlers[event_key] = []
        
        self.message_handlers[event_key].append(handler)
        logger.info(f"Added handler for event type: {event_key}")
    
    async def start_consuming(self) -> None:
        """Start consuming messages from subscribed topics."""
        if not self.consumer:
            await self.connect()
        
        self.is_running = True
        logger.info(f"Starting message consumption for group: {self.group_id}")
        
        try:
            while self.is_running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_message(message)
                
                # Commit offsets after successful processing
                try:
                    self.consumer.commit()
                except Exception as e:
                    logger.error(f"Error committing offsets: {e}")
                    
        except Exception as e:
            logger.error(f"Error in message consumption loop: {e}")
            raise
        finally:
            logger.info("Message consumption stopped")
    
    async def _process_message(self, message) -> None:
        """Process a single Kafka message."""
        try:
            # Deserialize message
            event_data = message.value
            event_type = event_data.get('event_type')
            
            self.stats['messages_consumed'] += 1
            
            # Find handlers for this event type
            handlers = self.message_handlers.get(event_type, [])
            
            if not handlers:
                logger.debug(f"No handlers found for event type: {event_type}")
                return
            
            # Execute all handlers for this event type
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data, message)
                    else:
                        handler(event_data, message)
                    
                    self.stats['messages_processed'] += 1
                    
                except Exception as e:
                    self.stats['processing_errors'] += 1
                    logger.error(f"Error in message handler: {e}")
                    
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error processing message: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            'is_running': self.is_running,
            'topics': self.topics,
            'group_id': self.group_id,
            'handlers': {k: len(v) for k, v in self.message_handlers.items()},
            'stats': self.stats.copy()
        }


class StreamProcessor:
    """Base class for stream processing applications."""
    
    def __init__(
        self,
        name: str,
        input_topics: List[str],
        output_topics: List[str],
        consumer_group: str,
        config: Optional[KafkaConfig] = None
    ):
        """Initialize stream processor.
        
        Args:
            name: Processor name
            input_topics: Topics to consume from
            output_topics: Topics to produce to
            consumer_group: Consumer group ID
            config: Kafka configuration
        """
        self.name = name
        self.input_topics = input_topics
        self.output_topics = output_topics
        self.consumer_group = consumer_group
        self.config = config or KafkaConfig()
        
        # Initialize Kafka clients
        self.consumer = KafkaConsumer(input_topics, consumer_group, self.config)
        self.producer = KafkaProducer(self.config)
        
        # Processing state
        self.is_running = False
        self.stats = {
            'messages_processed': 0,
            'messages_produced': 0,
            'processing_errors': 0,
            'start_time': None
        }
    
    async def start(self) -> None:
        """Start the stream processor."""
        log_function_call("start", processor=self.name)
        
        # Connect clients
        await self.consumer.connect()
        await self.producer.connect()
        
        # Add message handlers
        for input_topic in self.input_topics:
            self.consumer.add_handler('*', self._process_stream_message)  # Handle all messages
        
        # Start processing
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"Stream processor '{self.name}' started")
        
        # Start consuming messages
        await self.consumer.start_consuming()
    
    async def stop(self) -> None:
        """Stop the stream processor."""
        log_function_call("stop", processor=self.name)
        
        self.is_running = False
        
        # Disconnect clients
        await self.consumer.disconnect()
        await self.producer.disconnect()
        
        logger.info(f"Stream processor '{self.name}' stopped")
    
    async def _process_stream_message(self, event_data: Dict[str, Any], message) -> None:
        """Process incoming stream message."""
        try:
            # Call subclass implementation
            result = await self.process_message(event_data, message)
            
            # Send result to output topics if provided
            if result and self.output_topics:
                for output_topic in self.output_topics:
                    # Convert result to event if needed
                    if isinstance(result, BaseEvent):
                        await self.producer.send_event(output_topic, result)
                    elif isinstance(result, dict):
                        # Create a generic event
                        from .event_schemas import EventFactory, EventType
                        event = EventFactory.create_analytics_event(
                            metric_name="stream_processed",
                            metric_value=1,
                            metric_type="counter",
                            service=self.name,
                            source=f"stream_processor.{self.name}"
                        )
                        await self.producer.send_event(output_topic, event)
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error processing message in {self.name}: {e}")
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[Union[BaseEvent, Dict[str, Any]]]:
        """Process a single message. Must be implemented by subclasses.
        
        Args:
            event_data: Deserialized event data
            message: Raw Kafka message
            
        Returns:
            Optional result to send to output topics
        """
        raise NotImplementedError("Subclasses must implement process_message")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self.stats.copy()
        if stats['start_time']:
            runtime = (datetime.now() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['messages_per_second'] = stats['messages_processed'] / max(runtime, 1)
        
        return {
            'name': self.name,
            'input_topics': self.input_topics,
            'output_topics': self.output_topics,
            'consumer_group': self.consumer_group,
            'is_running': self.is_running,
            'stats': stats
        }