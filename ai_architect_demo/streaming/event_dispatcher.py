"""Event dispatcher for routing streaming events.

Provides intelligent event routing, filtering, and transformation
capabilities for the streaming architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union

from pydantic import BaseModel, Field

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.streaming.event_schemas import BaseEvent, EventType, EventFactory
from ai_architect_demo.streaming.kafka_client import KafkaConsumer, KafkaProducer

logger = get_logger(__name__)


class EventRoute(BaseModel):
    """Configuration for an event route."""
    
    name: str = Field(..., description="Route name")
    source_topics: List[str] = Field(..., description="Source topics to consume from")
    target_topics: List[str] = Field(..., description="Target topics to publish to")
    event_types: Optional[List[EventType]] = Field(None, description="Event types to route")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Event filters")
    transformations: List[str] = Field(default_factory=list, description="Transformation rules")
    priority: int = Field(1, description="Route priority (higher = more important)")
    enabled: bool = Field(True, description="Whether route is enabled")


class EventFilter:
    """Event filtering logic."""
    
    @staticmethod
    def matches_criteria(event_data: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if event matches filter criteria.
        
        Args:
            event_data: Event data to check
            criteria: Filter criteria
            
        Returns:
            True if event matches all criteria
        """
        for key, expected_value in criteria.items():
            # Handle nested keys with dot notation
            actual_value = EventFilter._get_nested_value(event_data, key)
            
            if actual_value is None:
                return False
            
            # Handle different comparison types
            if isinstance(expected_value, dict):
                if '$in' in expected_value:
                    if actual_value not in expected_value['$in']:
                        return False
                elif '$regex' in expected_value:
                    import re
                    if not re.match(expected_value['$regex'], str(actual_value)):
                        return False
                elif '$gt' in expected_value:
                    if actual_value <= expected_value['$gt']:
                        return False
                elif '$lt' in expected_value:
                    if actual_value >= expected_value['$lt']:
                        return False
                elif '$exists' in expected_value:
                    exists = actual_value is not None
                    if exists != expected_value['$exists']:
                        return False
            else:
                # Exact match
                if actual_value != expected_value:
                    return False
        
        return True
    
    @staticmethod
    def _get_nested_value(data: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value


class EventTransformer:
    """Event transformation logic."""
    
    @staticmethod
    def apply_transformations(
        event_data: Dict[str, Any],
        transformations: List[str]
    ) -> Dict[str, Any]:
        """Apply transformation rules to event data.
        
        Args:
            event_data: Event data to transform
            transformations: List of transformation rules
            
        Returns:
            Transformed event data
        """
        result = event_data.copy()
        
        for transformation in transformations:
            result = EventTransformer._apply_transformation(result, transformation)
        
        return result
    
    @staticmethod
    def _apply_transformation(data: Dict[str, Any], rule: str) -> Dict[str, Any]:
        """Apply single transformation rule."""
        if rule.startswith('add_field:'):
            # add_field:field_name=value
            _, field_spec = rule.split(':', 1)
            field_name, field_value = field_spec.split('=', 1)
            data[field_name] = field_value
            
        elif rule.startswith('remove_field:'):
            # remove_field:field_name
            _, field_name = rule.split(':', 1)
            data.pop(field_name, None)
            
        elif rule.startswith('rename_field:'):
            # rename_field:old_name=new_name
            _, field_spec = rule.split(':', 1)
            old_name, new_name = field_spec.split('=', 1)
            if old_name in data:
                data[new_name] = data.pop(old_name)
                
        elif rule.startswith('map_value:'):
            # map_value:field_name=old_value:new_value
            _, field_spec = rule.split(':', 1)
            parts = field_spec.split(':', 2)
            field_name, old_value, new_value = parts
            if data.get(field_name) == old_value:
                data[field_name] = new_value
                
        elif rule == 'add_timestamp':
            data['_transformed_at'] = datetime.now().isoformat()
            
        elif rule == 'add_routing_metadata':
            if '_routing' not in data:
                data['_routing'] = {}
            data['_routing']['processed_at'] = datetime.now().isoformat()
            data['_routing']['dispatcher'] = 'EventDispatcher'
        
        return data


class EventDispatcher:
    """Central event dispatcher for routing streaming events."""
    
    def __init__(self, name: str = "event_dispatcher"):
        """Initialize event dispatcher.
        
        Args:
            name: Dispatcher instance name
        """
        self.name = name
        self.routes: List[EventRoute] = []
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.producers: Dict[str, KafkaProducer] = {}
        self.is_running = False
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_routed': 0,
            'events_filtered': 0,
            'events_transformed': 0,
            'routing_errors': 0,
            'start_time': None
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def add_route(self, route: EventRoute) -> None:
        """Add an event route configuration.
        
        Args:
            route: Route configuration
        """
        self.routes.append(route)
        logger.info(f"Added event route: {route.name}")
        
        # Sort routes by priority (higher priority first)
        self.routes.sort(key=lambda r: r.priority, reverse=True)
    
    def add_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Add custom event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        event_key = event_type.value
        if event_key not in self.event_handlers:
            self.event_handlers[event_key] = []
        
        self.event_handlers[event_key].append(handler)
        logger.info(f"Added event handler for {event_type}")
    
    async def start(self) -> None:
        """Start the event dispatcher."""
        log_function_call("start", dispatcher=self.name)
        
        if not self.routes:
            logger.warning("No routes configured for event dispatcher")
            return
        
        # Collect all unique topics
        consumer_topics = set()
        producer_topics = set()
        
        for route in self.routes:
            if route.enabled:
                consumer_topics.update(route.source_topics)
                producer_topics.update(route.target_topics)
        
        # Create consumers for source topics
        for topic in consumer_topics:
            consumer_group = f"{self.name}_consumer_{topic}"
            consumer = KafkaConsumer([topic], consumer_group)
            
            # Add message handler
            consumer.add_handler('*', self._handle_event)
            
            await consumer.connect()
            self.consumers[topic] = consumer
        
        # Create producers for target topics
        for topic in producer_topics:
            producer = KafkaProducer()
            await producer.connect()
            self.producers[topic] = producer
        
        # Start consuming
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"Event dispatcher '{self.name}' started with {len(self.routes)} routes")
        
        # Start all consumers
        tasks = []
        for consumer in self.consumers.values():
            task = asyncio.create_task(consumer.start_consuming())
            tasks.append(task)
        
        # Wait for all consumers to finish
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self) -> None:
        """Stop the event dispatcher."""
        log_function_call("stop", dispatcher=self.name)
        
        self.is_running = False
        
        # Stop all consumers
        for consumer in self.consumers.values():
            await consumer.disconnect()
        
        # Stop all producers
        for producer in self.producers.values():
            await producer.disconnect()
        
        logger.info(f"Event dispatcher '{self.name}' stopped")
    
    async def _handle_event(self, event_data: Dict[str, Any], message) -> None:
        """Handle incoming event and route it according to rules."""
        self.stats['events_received'] += 1
        
        try:
            # Get source topic from Kafka message
            source_topic = message.topic
            event_type = event_data.get('event_type')
            
            logger.debug(f"Received event {event_type} from {source_topic}")
            
            # Find matching routes
            matching_routes = self._find_matching_routes(event_data, source_topic)
            
            if not matching_routes:
                logger.debug(f"No routes found for event {event_type} from {source_topic}")
                return
            
            # Process each matching route
            for route in matching_routes:
                await self._process_route(route, event_data, source_topic)
            
            # Execute custom event handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_data, message)
                        else:
                            handler(event_data, message)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
            
        except Exception as e:
            self.stats['routing_errors'] += 1
            logger.error(f"Error handling event: {e}")
    
    def _find_matching_routes(self, event_data: Dict[str, Any], source_topic: str) -> List[EventRoute]:
        """Find routes that match the given event."""
        matching_routes = []
        
        for route in self.routes:
            if not route.enabled:
                continue
            
            # Check if source topic matches
            if source_topic not in route.source_topics:
                continue
            
            # Check event type filter
            event_type = event_data.get('event_type')
            if route.event_types:
                event_type_enum = EventType(event_type) if event_type else None
                if event_type_enum not in route.event_types:
                    continue
            
            # Check custom filters
            if route.filters and not EventFilter.matches_criteria(event_data, route.filters):
                continue
            
            matching_routes.append(route)
        
        return matching_routes
    
    async def _process_route(self, route: EventRoute, event_data: Dict[str, Any], source_topic: str) -> None:
        """Process event through a specific route."""
        try:
            # Apply transformations
            transformed_data = event_data
            if route.transformations:
                transformed_data = EventTransformer.apply_transformations(
                    event_data, route.transformations
                )
                self.stats['events_transformed'] += 1
            
            # Send to target topics
            for target_topic in route.target_topics:
                # Get producer for target topic
                producer = self.producers.get(target_topic)
                if not producer:
                    logger.error(f"No producer configured for topic: {target_topic}")
                    continue
                
                # Create event from transformed data
                try:
                    # Try to create proper event type
                    event_type = EventType(transformed_data.get('event_type', 'system_health'))
                    event = EventFactory.create_from_dict(transformed_data, event_type)
                except (ValueError, KeyError):
                    # Fallback to generic system health event
                    event = EventFactory.create_system_health_event(
                        component="event_dispatcher",
                        status="info",
                        message=f"Routed event from {source_topic} via {route.name}",
                        source="event_dispatcher"
                    )
                
                # Send event
                success = await producer.send_event(
                    topic=target_topic,
                    event=event,
                    key=transformed_data.get('_routing_key')
                )
                
                if success:
                    self.stats['events_routed'] += 1
                    logger.debug(f"Event routed from {source_topic} to {target_topic} via {route.name}")
                else:
                    self.stats['routing_errors'] += 1
                    logger.error(f"Failed to route event to {target_topic}")
        
        except Exception as e:
            self.stats['routing_errors'] += 1
            logger.error(f"Error processing route {route.name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime = (datetime.now() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['events_per_second'] = stats['events_received'] / max(runtime, 1)
        
        return {
            'name': self.name,
            'is_running': self.is_running,
            'routes': len(self.routes),
            'active_routes': len([r for r in self.routes if r.enabled]),
            'consumers': len(self.consumers),
            'producers': len(self.producers),
            'stats': stats
        }


class PredefinedRoutes:
    """Factory for common event routing configurations."""
    
    @staticmethod
    def create_document_processing_routes() -> List[EventRoute]:
        """Create routes for document processing pipeline."""
        return [
            # Route document events to processing pipeline
            EventRoute(
                name="document_to_processing",
                source_topics=["documents"],
                target_topics=["document_processing", "analytics"],
                event_types=[EventType.DOCUMENT_PROCESSED, EventType.DOCUMENT_UPLOADED],
                transformations=["add_routing_metadata"],
                priority=10
            ),
            
            # Route processing results to notifications
            EventRoute(
                name="processing_to_notifications",
                source_topics=["document_processing"],
                target_topics=["notifications"],
                event_types=[EventType.DOCUMENT_PROCESSED],
                filters={"status": {"$in": ["completed", "failed"]}},
                transformations=[
                    "add_field:notification_type=processing_complete",
                    "add_timestamp"
                ],
                priority=8
            )
        ]
    
    @staticmethod
    def create_analytics_routes() -> List[EventRoute]:
        """Create routes for analytics pipeline."""
        return [
            # Route all events to analytics
            EventRoute(
                name="all_to_analytics",
                source_topics=["documents", "tasks", "agents", "notifications"],
                target_topics=["analytics", "metrics"],
                transformations=["add_routing_metadata"],
                priority=5
            ),
            
            # Route high-priority analytics to alerts
            EventRoute(
                name="analytics_to_alerts",
                source_topics=["analytics"],
                target_topics=["alerts"],
                event_types=[EventType.ANALYTICS_METRIC_UPDATE],
                filters={"metadata.priority": {"$in": ["high", "critical"]}},
                transformations=["add_field:alert_type=analytics_threshold"],
                priority=15
            )
        ]
    
    @staticmethod
    def create_agent_monitoring_routes() -> List[EventRoute]:
        """Create routes for agent monitoring."""
        return [
            # Route agent events to monitoring
            EventRoute(
                name="agents_to_monitoring",
                source_topics=["agents"],
                target_topics=["agent_monitoring", "analytics"],
                event_types=[EventType.AGENT_STATUS_CHANGED, EventType.AGENT_PERFORMANCE_UPDATE],
                transformations=["add_routing_metadata"],
                priority=12
            ),
            
            # Route agent errors to alerts
            EventRoute(
                name="agent_errors_to_alerts",
                source_topics=["agents"],
                target_topics=["alerts", "notifications"],
                event_types=[EventType.AGENT_ERROR],
                transformations=[
                    "add_field:alert_type=agent_error",
                    "add_field:priority=high"
                ],
                priority=20
            )
        ]
    
    @staticmethod
    def create_all_standard_routes() -> List[EventRoute]:
        """Create all standard routing configurations."""
        routes = []
        routes.extend(PredefinedRoutes.create_document_processing_routes())
        routes.extend(PredefinedRoutes.create_analytics_routes())
        routes.extend(PredefinedRoutes.create_agent_monitoring_routes())
        return routes