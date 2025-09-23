# Streaming Architecture Documentation

## Overview

The streaming architecture provides real-time event processing capabilities for the AI architecture demo system. Built on Apache Kafka, it enables real-time document processing, agent monitoring, analytics, and notifications.

## Architecture Components

### 1. Event System (`event_schemas.py`)

#### Core Event Types
- **DocumentEvent**: Document upload, processing, analysis
- **TaskEvent**: Task creation, execution, completion
- **AgentEvent**: Agent activities, task assignments, results
- **AnalyticsEvent**: Metrics, performance data, system stats
- **NotificationEvent**: User notifications, alerts, updates
- **QualityEvent**: Quality assurance checks and results
- **BusinessInsightEvent**: Business intelligence findings
- **SystemHealthEvent**: System status and health monitoring

#### EventFactory
Centralized factory for creating consistent, validated events:

```python
from ai_architect_demo.streaming import EventFactory, EventType

# Create document event
event = EventFactory.create_document_event(
    document_id="doc_001",
    action="uploaded",
    status="pending",
    file_path="/uploads/document.pdf",
    source="document_service"
)

# Create task event
task_event = EventFactory.create_task_event(
    task_id="task_001",
    task_type="document_analysis",
    action="created",
    status="pending",
    source="task_manager"
)
```

### 2. Kafka Client (`kafka_client.py`)

#### KafkaProducer
High-level producer with error handling and retry logic:

```python
from ai_architect_demo.streaming import KafkaProducer

producer = KafkaProducer()
await producer.connect()

# Send single event
success = await producer.send_event("documents", event)

# Send batch
results = await producer.send_batch("documents", events)

await producer.disconnect()
```

#### KafkaConsumer
Robust consumer with message handling and offset management:

```python
from ai_architect_demo.streaming import KafkaConsumer

consumer = KafkaConsumer(
    topics=["documents", "tasks"], 
    group_id="processor_group"
)

# Add message handlers
consumer.add_handler(EventType.DOCUMENT_UPLOADED, handle_document)
consumer.add_handler(EventType.TASK_CREATED, handle_task)

await consumer.connect()
await consumer.start_consuming()
```

#### StreamProcessor Base Class
Framework for building custom stream processors:

```python
from ai_architect_demo.streaming import StreamProcessor

class CustomProcessor(StreamProcessor):
    def __init__(self):
        super().__init__(
            name="custom_processor",
            input_topics=["input_topic"],
            output_topics=["output_topic"],
            consumer_group="custom_group"
        )
    
    async def process_message(self, event_data, message):
        # Custom processing logic
        result = await self.custom_logic(event_data)
        return result
```

### 3. Event Dispatcher (`event_dispatcher.py`)

Intelligent routing and transformation of events between topics:

```python
from ai_architect_demo.streaming import EventDispatcher, EventRoute

dispatcher = EventDispatcher("main_dispatcher")

# Add routing rule
route = EventRoute(
    name="documents_to_processing",
    source_topics=["documents"],
    target_topics=["processing", "analytics"],
    event_types=[EventType.DOCUMENT_UPLOADED],
    filters={"status": "pending"},
    transformations=["add_routing_metadata"],
    priority=10
)

dispatcher.add_route(route)
await dispatcher.start()
```

#### Predefined Routes
Common routing patterns:

```python
from ai_architect_demo.streaming import PredefinedRoutes

# Get standard routes
doc_routes = PredefinedRoutes.create_document_processing_routes()
analytics_routes = PredefinedRoutes.create_analytics_routes()
agent_routes = PredefinedRoutes.create_agent_monitoring_routes()

for route in doc_routes + analytics_routes + agent_routes:
    dispatcher.add_route(route)
```

### 4. Stream Processors (`stream_processors.py`)

#### Specialized Processors

**DocumentStreamProcessor**
- Processes document upload and analysis events
- Triggers automatic document analysis workflows
- Generates processing analytics

**TaskStreamProcessor**
- Manages task lifecycle events
- Tracks task execution metrics
- Handles task completion notifications

**AgentStreamProcessor**
- Monitors agent activities and performance
- Tracks agent task assignments and results
- Generates agent health metrics

**AnalyticsStreamProcessor**
- Aggregates metrics across time windows
- Checks threshold-based alerts
- Generates performance summaries

### 5. Real-time Monitoring (`real_time_monitor.py`)

Comprehensive monitoring and alerting system:

```python
from ai_architect_demo.streaming import RealTimeMonitor, AlertRule

monitor = RealTimeMonitor("system_monitor")

# Add custom alert rule
alert_rule = AlertRule(
    name="high_error_rate",
    metric="error_rate",
    condition="gt",
    threshold=5.0,
    severity="high",
    cooldown=300
)

monitor.add_alert_rule(alert_rule)
await monitor.start()

# Get monitoring dashboard
dashboard = monitor.get_dashboard_data()
```

#### Health Monitoring
- Component health checks
- System performance metrics
- Alert rule evaluation
- Real-time statistics

### 6. Agent Integration (`agent_integration.py`)

Seamless integration with the existing agentic AI system:

```python
from ai_architect_demo.streaming import create_integrated_streaming_system
from ai_architect_demo.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
streaming_components = await create_integrated_streaming_system(orchestrator)

# Components include:
# - Agent streaming integration
# - Agent stream processor  
# - Notification service
# - Real-time monitor
# - Event dispatcher
```

#### Features
- Automatic task routing to appropriate agents
- Real-time agent activity monitoring
- Agent error handling and notifications
- Streaming task notifications

### 7. Demo System (`demo_streaming.py`)

Complete demonstration of streaming capabilities:

```python
from ai_architect_demo.streaming import run_streaming_demo

# Run complete demo
results = await run_streaming_demo()

# Results include:
# - Event generation statistics
# - Processing metrics
# - Component health status
# - Alert summaries
```

## Usage Examples

### Basic Event Streaming

```python
import asyncio
from ai_architect_demo.streaming import (
    KafkaProducer, EventFactory, EventType
)

async def basic_streaming():
    producer = KafkaProducer()
    await producer.connect()
    
    # Create and send document event
    event = EventFactory.create_document_event(
        document_id="example_doc",
        action="uploaded",
        status="pending",
        file_path="/uploads/example.pdf",
        source="api_service"
    )
    
    success = await producer.send_event("documents", event)
    print(f"Event sent: {success}")
    
    await producer.disconnect()

asyncio.run(basic_streaming())
```

### Stream Processing Pipeline

```python
import asyncio
from ai_architect_demo.streaming import (
    DocumentStreamProcessor, TaskStreamProcessor,
    EventDispatcher, PredefinedRoutes
)

async def setup_pipeline():
    # Create processors
    doc_processor = DocumentStreamProcessor()
    task_processor = TaskStreamProcessor()
    
    # Create dispatcher with routing
    dispatcher = EventDispatcher("pipeline_dispatcher")
    routes = PredefinedRoutes.create_document_processing_routes()
    
    for route in routes:
        dispatcher.add_route(route)
    
    # Start all components
    await asyncio.gather(
        doc_processor.start(),
        task_processor.start(),
        dispatcher.start()
    )

asyncio.run(setup_pipeline())
```

### Agent Integration

```python
import asyncio
from ai_architect_demo.streaming import AgentStreamingIntegration
from ai_architect_demo.agents.orchestrator import AgentOrchestrator

async def setup_agent_streaming():
    orchestrator = AgentOrchestrator()
    integration = AgentStreamingIntegration(orchestrator)
    
    await integration.start()
    
    # Integration will automatically:
    # - Route streaming tasks to agents
    # - Monitor agent activities
    # - Send real-time notifications
    # - Generate analytics events

asyncio.run(setup_agent_streaming())
```

### Real-time Monitoring

```python
import asyncio
from ai_architect_demo.streaming import RealTimeMonitor, AlertRule

async def setup_monitoring():
    monitor = RealTimeMonitor("production_monitor")
    
    # Add custom alert rules
    rules = [
        AlertRule(
            name="high_latency",
            metric="avg_latency_ms", 
            condition="gt",
            threshold=1000.0,
            severity="medium"
        ),
        AlertRule(
            name="error_spike",
            metric="error_rate",
            condition="gt", 
            threshold=10.0,
            severity="high"
        )
    ]
    
    for rule in rules:
        monitor.add_alert_rule(rule)
    
    await monitor.start()
    
    # Get real-time dashboard
    dashboard = monitor.get_dashboard_data()
    print(f"Active alerts: {dashboard['active_alerts']}")

asyncio.run(setup_monitoring())
```

## Configuration

### Kafka Configuration

```python
from ai_architect_demo.streaming import KafkaConfig

# Custom Kafka configuration
config = KafkaConfig(
    bootstrap_servers="localhost:9092",
    security_protocol="PLAINTEXT",
    # Additional producer settings
    acks='all',
    retries=3,
    batch_size=16384,
    # Additional consumer settings  
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    max_poll_records=100
)

producer = KafkaProducer(config)
consumer = KafkaConsumer(["topic"], "group_id", config)
```

### Demo Configuration

```python
from ai_architect_demo.streaming import StreamingDemoConfig

config = StreamingDemoConfig(
    demo_duration_minutes=5,
    event_generation_rate=3.0,
    enable_monitoring=True,
    simulate_errors=True,
    error_rate_percent=2.0,
    # Event distribution
    document_events_percent=40.0,
    task_events_percent=30.0,
    agent_events_percent=20.0,
    analytics_events_percent=10.0
)
```

## Topics and Event Flow

### Topic Structure
```
documents/           # Document events
├── document_uploaded
├── document_processed
└── document_analyzed

tasks/              # Task events  
├── task_created
├── task_started
├── task_completed
└── task_failed

agents/             # Agent events
├── agent_task_started
├── agent_task_completed
├── agent_error
└── agent_health

analytics/          # Analytics events
├── metrics
├── performance_data
└── system_stats

notifications/      # Notification events
├── user_notifications
├── alerts
└── system_updates

system_health/      # Health monitoring
├── component_health
├── performance_metrics
└── alert_events
```

### Event Flow Diagram
```
Document Upload → [documents] → DocumentProcessor
                              ↓
                          [processing] → AnalyticsProcessor
                              ↓              ↓
Task Creation → [tasks] → TaskProcessor → [analytics] → Monitor
                              ↓              ↓
Agent Activity → [agents] → AgentProcessor → [alerts] → Notifications
```

## Monitoring and Observability

### Metrics Collected
- **Event Throughput**: Events per second by topic/type
- **Processing Latency**: End-to-end processing times
- **Error Rates**: Error percentages by component
- **Component Health**: Service availability and status
- **Resource Usage**: CPU, memory, and network metrics

### Alert Types
- **Threshold Alerts**: Metric-based alerting
- **Health Alerts**: Component availability issues
- **Error Alerts**: High error rates or critical failures
- **Performance Alerts**: Latency or throughput issues

### Dashboard Data
```python
{
    "timestamp": "2024-01-01T12:00:00Z",
    "uptime_seconds": 3600,
    "component_health": {
        "kafka": {"status": "healthy", "message": "All brokers responding"},
        "document_processor": {"status": "healthy", "message": "Processing normally"},
        "agent_orchestrator": {"status": "healthy", "message": "All agents active"}
    },
    "metrics": {
        "events_per_second": 25.5,
        "avg_latency_ms": 125.3,
        "error_rate": 0.5,
        "active_tasks": 15
    },
    "alerts": {
        "active_count": 0,
        "total_triggered": 3,
        "last_alert": "2024-01-01T11:45:00Z"
    }
}
```

## Performance Considerations

### Throughput Optimization
- Batch event sending for high volume scenarios
- Configurable consumer poll sizes and timeouts
- Parallel processing across multiple partitions
- Connection pooling and reuse

### Reliability Features
- Automatic retry logic with exponential backoff
- Dead letter queue handling for failed messages
- Offset management and replay capabilities
- Health checks and automatic recovery

### Scalability
- Horizontal scaling through consumer groups
- Partitioned topics for parallel processing
- Load balancing across multiple instances
- Dynamic topic and partition management

## Testing

### Integration Tests
Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest ai_architect_demo/streaming/integration_test.py -v

# Run basic tests only
python ai_architect_demo/streaming/integration_test.py basic

# Run performance tests
python -m pytest ai_architect_demo/streaming/integration_test.py::test_streaming_performance -v
```

### Demo Execution
```bash
# Run complete demo
python -c "
import asyncio
from ai_architect_demo.streaming import run_streaming_demo
results = asyncio.run(run_streaming_demo())
print(results['summary'])
"
```

## Troubleshooting

### Common Issues

1. **Kafka Connection Failed**
   - Verify Kafka is running: `docker-compose ps`
   - Check bootstrap servers configuration
   - Ensure network connectivity

2. **Events Not Processing**
   - Check consumer group subscriptions
   - Verify topic existence and permissions
   - Review consumer lag metrics

3. **High Latency**
   - Increase batch sizes for producers
   - Optimize consumer poll settings
   - Check network and broker performance

4. **Memory Issues**
   - Configure appropriate buffer sizes
   - Implement proper cleanup and resource management
   - Monitor metrics history retention

### Logging
Enable debug logging:

```python
import logging
logging.getLogger('ai_architect_demo.streaming').setLevel(logging.DEBUG)
```

### Monitoring Commands
```python
# Get component statistics
stats = component.get_stats()

# Check health status  
health = monitor.get_dashboard_data()

# View active alerts
alerts = monitor.active_alerts

# Get processing metrics
metrics = processor.get_metric_summary()
```

## Future Enhancements

### Planned Features
- Stream processing with windowed aggregations
- Advanced ML-based anomaly detection
- Multi-cluster Kafka support
- Enhanced visualization dashboards
- Integration with external monitoring systems

### Performance Improvements
- Schema registry integration
- Avro/Protobuf serialization
- Exactly-once processing semantics
- Advanced partition strategies
- Streaming SQL capabilities

---

This streaming architecture provides a robust foundation for real-time event processing in the AI architecture demo system. It seamlessly integrates with the existing agentic AI components while providing comprehensive monitoring, alerting, and analytics capabilities.