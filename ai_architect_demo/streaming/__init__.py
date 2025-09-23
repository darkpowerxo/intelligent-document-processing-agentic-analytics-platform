"""Streaming architecture module for real-time event processing.

This module provides Kafka-based event streaming capabilities
for real-time document processing and analytics.
"""

__version__ = "1.0.0"

# Import main components for easy access
from .event_schemas import (
    BaseEvent, EventType, EventFactory,
    DocumentEvent, TaskEvent, AgentEvent, AnalyticsEvent,
    NotificationEvent, QualityEvent, BusinessInsightEvent, SystemHealthEvent
)

from .kafka_client import (
    KafkaConfig, KafkaProducer, KafkaConsumer, StreamProcessor
)

from .event_dispatcher import (
    EventDispatcher, EventRoute, PredefinedRoutes
)

from .stream_processors import (
    DocumentStreamProcessor, TaskStreamProcessor,
    AgentStreamProcessor, AnalyticsStreamProcessor, NotificationStreamProcessor
)

from .real_time_monitor import (
    RealTimeMonitor, HealthStatus, MetricSnapshot, SystemMetrics, AlertRule
)

from .agent_integration import (
    AgentStreamingIntegration, StreamingAgentProcessor, 
    StreamingNotificationService, create_integrated_streaming_system
)

from .demo_streaming import (
    StreamingDemoConfig, StreamingEventSimulator, 
    StreamingDemoOrchestrator, run_streaming_demo
)

__all__ = [
    # Event schemas
    'BaseEvent', 'EventType', 'EventFactory',
    'DocumentEvent', 'TaskEvent', 'AgentEvent', 'AnalyticsEvent',
    'NotificationEvent', 'QualityEvent', 'BusinessInsightEvent', 'SystemHealthEvent',
    
    # Kafka client
    'KafkaConfig', 'KafkaProducer', 'KafkaConsumer', 'StreamProcessor',
    
    # Event dispatcher
    'EventDispatcher', 'EventRoute', 'PredefinedRoutes',
    
    # Stream processors
    'DocumentStreamProcessor', 'TaskStreamProcessor',
    'AgentStreamProcessor', 'AnalyticsStreamProcessor', 'NotificationStreamProcessor',
    
    # Real-time monitoring
    'RealTimeMonitor', 'HealthStatus', 'MetricSnapshot', 'SystemMetrics', 'AlertRule',
    
    # Agent integration
    'AgentStreamingIntegration', 'StreamingAgentProcessor', 
    'StreamingNotificationService', 'create_integrated_streaming_system',
    
    # Demo system
    'StreamingDemoConfig', 'StreamingEventSimulator', 
    'StreamingDemoOrchestrator', 'run_streaming_demo'
]