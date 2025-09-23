"""Real-time monitoring and observability for the streaming system.

Provides live monitoring, health checks, and performance metrics
for all streaming components.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.streaming.event_schemas import BaseEvent, EventType, EventFactory
from ai_architect_demo.streaming.kafka_client import KafkaConsumer, KafkaProducer

logger = get_logger(__name__)


class HealthStatus(BaseModel):
    """Health status for a component."""
    
    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, warning, critical, down)")
    message: str = Field("", description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status in ["healthy", "warning"]


class MetricSnapshot(BaseModel):
    """Snapshot of a metric at a point in time."""
    
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.now)
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: str = Field("", description="Metric unit")


class SystemMetrics(BaseModel):
    """System-wide metrics snapshot."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    kafka_metrics: Dict[str, Any] = Field(default_factory=dict)
    processor_metrics: Dict[str, Any] = Field(default_factory=dict)
    dispatcher_metrics: Dict[str, Any] = Field(default_factory=dict)
    event_throughput: Dict[str, float] = Field(default_factory=dict)
    error_rates: Dict[str, float] = Field(default_factory=dict)
    latency_metrics: Dict[str, float] = Field(default_factory=dict)


class AlertRule(BaseModel):
    """Alert rule configuration."""
    
    name: str = Field(..., description="Rule name")
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (gt, lt, eq)")
    threshold: float = Field(..., description="Alert threshold")
    severity: str = Field("medium", description="Alert severity")
    cooldown: int = Field(300, description="Cooldown period in seconds")
    enabled: bool = Field(True, description="Whether rule is enabled")
    
    # Runtime state
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, value: float) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown:
                return False
        
        # Check condition
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        
        return False


class RealTimeMonitor:
    """Real-time monitoring system for streaming infrastructure."""
    
    def __init__(self, name: str = "stream_monitor"):
        """Initialize real-time monitor.
        
        Args:
            name: Monitor instance name
        """
        self.name = name
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Monitoring components
        self.kafka_producer = KafkaProducer()
        self.kafka_consumer = KafkaConsumer(
            topics=["analytics", "system_health", "alerts"],
            group_id="stream_monitor_consumer"
        )
        
        # Health tracking
        self.component_health: Dict[str, HealthStatus] = {}
        self.health_check_interval = 30  # seconds
        
        # Metrics collection
        self.metrics_history: List[SystemMetrics] = []
        self.metrics_retention_hours = 24
        self.metrics_collection_interval = 10  # seconds
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}
        
        # Performance tracking
        self.event_counters: Dict[str, int] = {}
        self.error_counters: Dict[str, int] = {}
        self.latency_samples: Dict[str, List[float]] = {}
        
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric="error_rate",
                condition="gt",
                threshold=5.0,
                severity="high",
                cooldown=300
            ),
            AlertRule(
                name="low_throughput",
                metric="events_per_second",
                condition="lt",
                threshold=1.0,
                severity="medium",
                cooldown=600
            ),
            AlertRule(
                name="high_latency",
                metric="avg_latency_ms",
                condition="gt",
                threshold=1000.0,
                severity="medium",
                cooldown=300
            ),
            AlertRule(
                name="component_down",
                metric="component_health",
                condition="eq",
                threshold=0.0,  # 0 = down
                severity="critical",
                cooldown=60
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    async def start(self) -> None:
        """Start the monitoring system."""
        log_function_call("start", monitor=self.name)
        
        # Connect Kafka clients
        await self.kafka_producer.connect()
        await self.kafka_consumer.connect()
        
        # Setup event handlers
        self.kafka_consumer.add_handler(EventType.ANALYTICS_METRIC_UPDATE, self._handle_analytics_event)
        self.kafka_consumer.add_handler(EventType.SYSTEM_HEALTH, self._handle_health_event)
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Real-time monitor '{self.name}' started")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self.kafka_consumer.start_consuming())
        ]
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        log_function_call("stop", monitor=self.name)
        
        self.is_running = False
        
        # Disconnect clients
        await self.kafka_consumer.disconnect()
        await self.kafka_producer.disconnect()
        
        logger.info(f"Real-time monitor '{self.name}' stopped")
    
    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_running:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Publish metrics
                await self._publish_metrics(metrics)
                
                await asyncio.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _health_check_loop(self) -> None:
        """Health check loop for all components."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _alert_processing_loop(self) -> None:
        """Alert processing and evaluation loop."""
        while self.is_running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(10)  # Check alerts every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        metrics = SystemMetrics()
        
        # Kafka metrics
        if hasattr(self.kafka_producer, 'get_stats'):
            producer_stats = self.kafka_producer.get_stats()
            metrics.kafka_metrics['producer'] = producer_stats
        
        if hasattr(self.kafka_consumer, 'get_stats'):
            consumer_stats = self.kafka_consumer.get_stats()
            metrics.kafka_metrics['consumer'] = consumer_stats
        
        # Event throughput calculation
        current_time = datetime.now()
        time_window = timedelta(minutes=1)
        
        for event_type, count in self.event_counters.items():
            # Calculate events per second (simplified)
            metrics.event_throughput[event_type] = count / 60.0  # Rough approximation
        
        # Error rates
        for component, error_count in self.error_counters.items():
            total_events = sum(self.event_counters.values()) or 1
            metrics.error_rates[component] = (error_count / total_events) * 100
        
        # Latency metrics
        for component, samples in self.latency_samples.items():
            if samples:
                metrics.latency_metrics[f"{component}_avg"] = sum(samples) / len(samples)
                metrics.latency_metrics[f"{component}_max"] = max(samples)
                metrics.latency_metrics[f"{component}_min"] = min(samples)
        
        return metrics
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        # Check Kafka connectivity
        kafka_health = await self._check_kafka_health()
        self.component_health['kafka'] = kafka_health
        
        # Check component availability (placeholder)
        # In a real implementation, you would ping each component
        components = ['document_processor', 'task_processor', 'agent_processor', 'analytics_processor']
        
        for component in components:
            health = HealthStatus(
                component=component,
                status="healthy",  # Placeholder - would actually check
                message="Component responsive"
            )
            self.component_health[component] = health
        
        # Publish health status
        for component, health in self.component_health.items():
            health_event = EventFactory.create_system_health_event(
                component=component,
                status=health.status,
                message=health.message,
                metadata=health.metadata,
                source="real_time_monitor"
            )
            
            await self.kafka_producer.send_event("system_health", health_event)
    
    async def _check_kafka_health(self) -> HealthStatus:
        """Check Kafka cluster health."""
        try:
            # Simple health check by sending a test message
            test_event = EventFactory.create_system_health_event(
                component="kafka_health_check",
                status="healthy",
                message="Health check ping",
                source="real_time_monitor"
            )
            
            success = await self.kafka_producer.send_event("system_health", test_event)
            
            if success:
                return HealthStatus(
                    component="kafka",
                    status="healthy",
                    message="Kafka cluster responsive"
                )
            else:
                return HealthStatus(
                    component="kafka",
                    status="warning",
                    message="Message send failed"
                )
                
        except Exception as e:
            return HealthStatus(
                component="kafka",
                status="critical",
                message=f"Kafka health check failed: {str(e)}"
            )
    
    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against current metrics."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        for rule in self.alert_rules:
            try:
                # Get metric value
                metric_value = self._get_metric_value(latest_metrics, rule.metric)
                if metric_value is None:
                    continue
                
                # Check if alert should trigger
                if rule.should_trigger(metric_value):
                    await self._trigger_alert(rule, metric_value)
                    rule.last_triggered = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _get_metric_value(self, metrics: SystemMetrics, metric_name: str) -> Optional[float]:
        """Extract metric value from metrics snapshot."""
        if metric_name == "error_rate":
            return sum(metrics.error_rates.values()) / max(len(metrics.error_rates), 1)
        elif metric_name == "events_per_second":
            return sum(metrics.event_throughput.values())
        elif metric_name == "avg_latency_ms":
            latency_values = [v for k, v in metrics.latency_metrics.items() if 'avg' in k]
            return sum(latency_values) / max(len(latency_values), 1) if latency_values else 0
        elif metric_name == "component_health":
            unhealthy_components = len([h for h in self.component_health.values() if not h.is_healthy()])
            return float(unhealthy_components)
        
        return None
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float) -> None:
        """Trigger an alert."""
        alert_event = EventFactory.create_notification_event(
            notification_type="alert",
            recipient="system_admin",
            message=f"Alert: {rule.name} - {rule.metric} = {metric_value} (threshold: {rule.threshold})",
            priority=rule.severity,
            metadata={
                "rule_name": rule.name,
                "metric": rule.metric,
                "value": metric_value,
                "threshold": rule.threshold,
                "condition": rule.condition
            },
            source="real_time_monitor"
        )
        
        await self.kafka_producer.send_event("alerts", alert_event)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.metric} = {metric_value}")
        
        # Track active alert
        self.active_alerts[rule.name] = datetime.now()
    
    async def _publish_metrics(self, metrics: SystemMetrics) -> None:
        """Publish metrics to analytics stream."""
        # Convert metrics to analytics events
        analytics_events = []
        
        # Event throughput metrics
        for event_type, throughput in metrics.event_throughput.items():
            event = EventFactory.create_analytics_event(
                metric_name=f"throughput.{event_type}",
                metric_value=throughput,
                metric_type="gauge",
                service="streaming_system",
                source="real_time_monitor"
            )
            analytics_events.append(event)
        
        # Error rate metrics
        for component, error_rate in metrics.error_rates.items():
            event = EventFactory.create_analytics_event(
                metric_name=f"error_rate.{component}",
                metric_value=error_rate,
                metric_type="gauge",
                service="streaming_system",
                source="real_time_monitor"
            )
            analytics_events.append(event)
        
        # Latency metrics
        for metric_name, latency in metrics.latency_metrics.items():
            event = EventFactory.create_analytics_event(
                metric_name=f"latency.{metric_name}",
                metric_value=latency,
                metric_type="gauge",
                service="streaming_system",
                source="real_time_monitor"
            )
            analytics_events.append(event)
        
        # Send all metrics events
        for event in analytics_events:
            await self.kafka_producer.send_event("analytics", event)
    
    async def _handle_analytics_event(self, event_data: Dict[str, Any], message) -> None:
        """Handle incoming analytics events."""
        metric_name = event_data.get('metric_name', '')
        metric_value = event_data.get('metric_value', 0)
        service = event_data.get('service', 'unknown')
        
        # Update event counters
        counter_key = f"{service}.{metric_name}"
        self.event_counters[counter_key] = self.event_counters.get(counter_key, 0) + 1
        
        # Track latency if it's a latency metric
        if 'latency' in metric_name or 'response_time' in metric_name:
            if counter_key not in self.latency_samples:
                self.latency_samples[counter_key] = []
            
            self.latency_samples[counter_key].append(metric_value)
            
            # Keep only recent samples (last 100)
            self.latency_samples[counter_key] = self.latency_samples[counter_key][-100:]
    
    async def _handle_health_event(self, event_data: Dict[str, Any], message) -> None:
        """Handle incoming health events."""
        component = event_data.get('component', 'unknown')
        status = event_data.get('status', 'unknown')
        message_text = event_data.get('message', '')
        
        # Update component health
        health = HealthStatus(
            component=component,
            status=status,
            message=message_text,
            metadata=event_data.get('metadata', {})
        )
        
        self.component_health[component] = health
        
        # Count errors
        if status in ['error', 'critical', 'down']:
            self.error_counters[component] = self.error_counters.get(component, 0) + 1
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics to prevent memory buildup."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'is_running': self.is_running,
            'component_health': {
                comp: {
                    'status': health.status,
                    'message': health.message,
                    'timestamp': health.timestamp.isoformat()
                }
                for comp, health in self.component_health.items()
            },
            'active_alerts': len(self.active_alerts),
            'alert_rules': len([r for r in self.alert_rules if r.enabled]),
            'metrics_count': len(self.metrics_history),
            'event_counters': self.event_counters.copy(),
            'error_counters': self.error_counters.copy(),
            'latest_metrics': self.metrics_history[-1].model_dump() if self.metrics_history else None
        }
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        for i, rule in enumerate(self.alert_rules):
            if rule.name == rule_name:
                del self.alert_rules[i]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False