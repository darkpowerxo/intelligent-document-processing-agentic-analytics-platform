"""Event schemas for streaming architecture.

Defines structured event types for Kafka streams using Pydantic models
to ensure consistent data formats across all streaming operations.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class EventType(Enum):
    """Event type classifications."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_ANALYSIS_COMPLETE = "document.analysis.complete"
    DOCUMENT_ERROR = "document.error"
    
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    
    AGENT_STATUS_CHANGED = "agent.status.changed"
    AGENT_PERFORMANCE_UPDATE = "agent.performance.update"
    
    ANALYTICS_METRIC_UPDATE = "analytics.metric.update"
    ANALYTICS_REPORT_GENERATED = "analytics.report.generated"
    
    USER_NOTIFICATION = "user.notification"
    SYSTEM_ALERT = "system.alert"
    
    QUALITY_CHECK_COMPLETE = "quality.check.complete"
    BUSINESS_INSIGHT_GENERATED = "business.insight.generated"


class Priority(Enum):
    """Event priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseEvent(BaseModel):
    """Base event model with common fields."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM
    source: str  # Service/component that generated the event
    correlation_id: Optional[str] = None  # For tracking related events
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class DocumentEvent(BaseEvent):
    """Events related to document processing."""
    document_id: str
    document_name: Optional[str] = None
    document_type: Optional[str] = None
    file_size: Optional[int] = None
    user_id: Optional[str] = None
    processing_stage: Optional[str] = None
    
    # Document-specific data
    content_preview: Optional[str] = None  # First 500 chars
    extracted_metadata: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, str]] = None
    
    # Processing metrics
    processing_time_ms: Optional[int] = None
    agent_id: Optional[str] = None


class TaskEvent(BaseEvent):
    """Events related to agent task processing."""
    task_id: str
    task_type: str
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None
    requester_id: Optional[str] = None
    
    # Task execution data
    task_data: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Performance metrics
    execution_time_ms: Optional[int] = None
    queue_time_ms: Optional[int] = None
    retry_count: Optional[int] = 0
    
    # Task relationships
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = Field(default_factory=list)


class AgentEvent(BaseEvent):
    """Events related to agent status and performance."""
    agent_id: str
    agent_name: str
    agent_role: str
    
    # Status information
    status: str  # idle, working, error, offline
    current_tasks: int = 0
    completed_tasks: int = 0
    error_count: int = 0
    
    # Performance metrics
    average_processing_time: Optional[float] = None
    success_rate: Optional[float] = None
    throughput_per_hour: Optional[float] = None
    
    # Resource utilization
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None


class AnalyticsEvent(BaseEvent):
    """Events for analytics and metrics updates."""
    metric_name: str
    metric_value: Union[int, float, str]
    metric_type: str  # counter, gauge, histogram, summary
    
    # Context information
    service: str
    environment: str = "production"
    tags: Dict[str, str] = Field(default_factory=dict)
    
    # Time-series data
    time_window: Optional[str] = None  # 1m, 5m, 1h, 1d
    aggregation: Optional[str] = None  # sum, avg, min, max, count
    
    # Additional metric data
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    threshold_alerts: List[Dict[str, Any]] = Field(default_factory=list)


class NotificationEvent(BaseEvent):
    """Events for user notifications and system alerts."""
    notification_type: str  # info, warning, error, success
    title: str
    message: str
    
    # Recipient information
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    channel: str = "system"  # email, sms, slack, system
    
    # Notification behavior
    is_persistent: bool = False
    auto_dismiss_after: Optional[int] = None  # seconds
    requires_acknowledgment: bool = False
    
    # Rich content
    action_buttons: List[Dict[str, str]] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Related entities
    related_document_id: Optional[str] = None
    related_task_id: Optional[str] = None
    related_agent_id: Optional[str] = None


class QualityEvent(BaseEvent):
    """Events related to quality assurance and validation."""
    validation_id: str
    subject_type: str  # document, task_result, agent_output
    subject_id: str
    
    # Quality metrics
    quality_score: float  # 0.0 - 1.0
    validation_criteria: List[str]
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str] = Field(default_factory=list)
    
    # Validation details
    validator_agent_id: Optional[str] = None
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    remediation_suggestions: List[str] = Field(default_factory=list)
    
    # Quality trends
    previous_quality_score: Optional[float] = None
    quality_trend: Optional[str] = None  # improving, declining, stable


class BusinessInsightEvent(BaseEvent):
    """Events for business intelligence insights and discoveries."""
    insight_id: str
    insight_type: str  # trend, anomaly, opportunity, risk
    confidence: float  # 0.0 - 1.0
    
    # Insight content
    title: str
    description: str
    key_findings: List[str]
    recommendations: List[str] = Field(default_factory=list)
    
    # Supporting data
    data_sources: List[str]
    analysis_period: Optional[str] = None
    metrics_analyzed: List[str] = Field(default_factory=list)
    
    # Business context
    business_impact: str  # low, medium, high, critical
    affected_areas: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    
    # Insight metadata
    generator_agent_id: str
    data_freshness: str  # real-time, hourly, daily
    validity_period: Optional[str] = None


class SystemHealthEvent(BaseEvent):
    """Events for system health monitoring."""
    component: str
    health_status: str  # healthy, warning, critical, down
    
    # Health metrics
    response_time_ms: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[float] = None
    
    # Resource metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None
    
    # Service-specific metrics
    active_connections: Optional[int] = None
    queue_depth: Optional[int] = None
    cache_hit_rate: Optional[float] = None
    
    # Alert information
    alert_level: str = "info"  # info, warning, critical
    alert_message: Optional[str] = None
    recovery_suggestions: List[str] = Field(default_factory=list)


# Event factory for creating events from different sources
class EventFactory:
    """Factory for creating different types of events."""
    
    @staticmethod
    def create_document_event(
        event_type: EventType,
        document_id: str,
        source: str,
        **kwargs
    ) -> DocumentEvent:
        """Create a document-related event."""
        return DocumentEvent(
            event_type=event_type,
            document_id=document_id,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_task_event(
        event_type: EventType,
        task_id: str,
        task_type: str,
        source: str,
        **kwargs
    ) -> TaskEvent:
        """Create a task-related event."""
        return TaskEvent(
            event_type=event_type,
            task_id=task_id,
            task_type=task_type,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_agent_event(
        event_type: EventType,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        source: str,
        **kwargs
    ) -> AgentEvent:
        """Create an agent-related event."""
        return AgentEvent(
            event_type=event_type,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_analytics_event(
        metric_name: str,
        metric_value: Union[int, float, str],
        metric_type: str,
        service: str,
        source: str,
        **kwargs
    ) -> AnalyticsEvent:
        """Create an analytics event."""
        return AnalyticsEvent(
            event_type=EventType.ANALYTICS_METRIC_UPDATE,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            service=service,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_notification_event(
        notification_type: str,
        title: str,
        message: str,
        source: str,
        **kwargs
    ) -> NotificationEvent:
        """Create a notification event."""
        return NotificationEvent(
            event_type=EventType.USER_NOTIFICATION,
            notification_type=notification_type,
            title=title,
            message=message,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_quality_event(
        validation_id: str,
        subject_type: str,
        subject_id: str,
        quality_score: float,
        source: str,
        **kwargs
    ) -> QualityEvent:
        """Create a quality assurance event."""
        return QualityEvent(
            event_type=EventType.QUALITY_CHECK_COMPLETE,
            validation_id=validation_id,
            subject_type=subject_type,
            subject_id=subject_id,
            quality_score=quality_score,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_business_insight_event(
        insight_id: str,
        insight_type: str,
        title: str,
        description: str,
        confidence: float,
        generator_agent_id: str,
        source: str,
        **kwargs
    ) -> BusinessInsightEvent:
        """Create a business insight event."""
        return BusinessInsightEvent(
            event_type=EventType.BUSINESS_INSIGHT_GENERATED,
            insight_id=insight_id,
            insight_type=insight_type,
            title=title,
            description=description,
            confidence=confidence,
            generator_agent_id=generator_agent_id,
            source=source,
            **kwargs
        )
    
    @staticmethod
    def create_system_health_event(
        component: str,
        health_status: str,
        source: str,
        **kwargs
    ) -> SystemHealthEvent:
        """Create a system health event."""
        return SystemHealthEvent(
            event_type=EventType.SYSTEM_ALERT,
            component=component,
            health_status=health_status,
            source=source,
            **kwargs
        )