"""Specialized stream processors for different event types.

Each processor handles specific types of events and implements
domain-specific processing logic.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.streaming.event_schemas import (
    BaseEvent, EventType, EventFactory,
    DocumentEvent, TaskEvent, AgentEvent, AnalyticsEvent
)
from ai_architect_demo.streaming.kafka_client import StreamProcessor

logger = get_logger(__name__)


class DocumentStreamProcessor(StreamProcessor):
    """Stream processor for document-related events."""
    
    def __init__(self, config=None):
        super().__init__(
            name="document_processor",
            input_topics=["documents"],
            output_topics=["document_processing", "analytics", "notifications"],
            consumer_group="document_stream_processor",
            config=config
        )
        
        # Document processing state
        self.processing_stats = {
            'documents_processed': 0,
            'documents_failed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[BaseEvent]:
        """Process document events."""
        try:
            event_type = EventType(event_data.get('event_type'))
            
            if event_type == EventType.DOCUMENT_UPLOADED:
                return await self._handle_document_upload(event_data)
            elif event_type == EventType.DOCUMENT_PROCESSED:
                return await self._handle_document_processed(event_data)
            else:
                logger.debug(f"Unhandled document event type: {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing document event: {e}")
            return EventFactory.create_system_health_event(
                component="document_processor",
                status="error",
                message=f"Failed to process document event: {str(e)}",
                source="document_stream_processor"
            )
    
    async def _handle_document_upload(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle document upload events."""
        document_id = event_data.get('document_id')
        file_path = event_data.get('file_path')
        
        logger.info(f"Processing uploaded document: {document_id}")
        
        # Simulate document analysis
        await asyncio.sleep(0.1)  # Simulated processing time
        
        # Create processing completion event
        result_event = EventFactory.create_document_event(
            document_id=document_id,
            action="processed",
            status="completed",
            file_path=file_path,
            metadata={
                "processed_by": "document_stream_processor",
                "processing_time": 0.1,
                "analysis_results": {
                    "pages": 5,
                    "word_count": 1250,
                    "language": "en",
                    "confidence": 0.95
                }
            },
            source="document_stream_processor"
        )
        
        self.processing_stats['documents_processed'] += 1
        return result_event
    
    async def _handle_document_processed(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle document processing completion events."""
        document_id = event_data.get('document_id')
        status = event_data.get('status', 'unknown')
        
        # Generate analytics event
        analytics_event = EventFactory.create_analytics_event(
            metric_name="document_processing_completed",
            metric_value=1,
            metric_type="counter",
            service="document_processor",
            metadata={
                "document_id": document_id,
                "status": status,
                "processor": "document_stream_processor"
            },
            source="document_stream_processor"
        )
        
        return analytics_event


class TaskStreamProcessor(StreamProcessor):
    """Stream processor for task-related events."""
    
    def __init__(self, config=None):
        super().__init__(
            name="task_processor",
            input_topics=["tasks"],
            output_topics=["task_monitoring", "analytics", "notifications"],
            consumer_group="task_stream_processor",
            config=config
        )
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[BaseEvent]:
        """Process task events."""
        try:
            event_type = EventType(event_data.get('event_type'))
            
            if event_type == EventType.TASK_CREATED:
                return await self._handle_task_created(event_data)
            elif event_type == EventType.TASK_STARTED:
                return await self._handle_task_started(event_data)
            elif event_type == EventType.TASK_COMPLETED:
                return await self._handle_task_completed(event_data)
            elif event_type == EventType.TASK_FAILED:
                return await self._handle_task_failed(event_data)
            else:
                logger.debug(f"Unhandled task event type: {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing task event: {e}")
            return None
    
    async def _handle_task_created(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle task creation events."""
        task_id = event_data.get('task_id')
        task_type = event_data.get('task_type', 'unknown')
        
        # Track new task
        self.active_tasks[task_id] = {
            'created_at': datetime.now(),
            'task_type': task_type,
            'status': 'created'
        }
        
        logger.info(f"New task created: {task_id} ({task_type})")
        
        # Create analytics event
        return EventFactory.create_analytics_event(
            metric_name="task_created",
            metric_value=1,
            metric_type="counter",
            service="task_processor",
            metadata={
                "task_id": task_id,
                "task_type": task_type
            },
            source="task_stream_processor"
        )
    
    async def _handle_task_started(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle task start events."""
        task_id = event_data.get('task_id')
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'running'
            self.active_tasks[task_id]['started_at'] = datetime.now()
        
        return EventFactory.create_analytics_event(
            metric_name="task_started",
            metric_value=1,
            metric_type="counter",
            service="task_processor",
            metadata={"task_id": task_id},
            source="task_stream_processor"
        )
    
    async def _handle_task_completed(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle task completion events."""
        task_id = event_data.get('task_id')
        
        # Calculate processing time
        processing_time = 0.0
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            if 'started_at' in task_info:
                processing_time = (datetime.now() - task_info['started_at']).total_seconds()
            
            # Move to completed
            del self.active_tasks[task_id]
        
        self.completed_tasks += 1
        
        return EventFactory.create_analytics_event(
            metric_name="task_completed",
            metric_value=1,
            metric_type="counter",
            service="task_processor",
            metadata={
                "task_id": task_id,
                "processing_time": processing_time
            },
            source="task_stream_processor"
        )
    
    async def _handle_task_failed(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle task failure events."""
        task_id = event_data.get('task_id')
        error = event_data.get('error', 'Unknown error')
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        self.failed_tasks += 1
        
        # Create high-priority analytics event
        return EventFactory.create_analytics_event(
            metric_name="task_failed",
            metric_value=1,
            metric_type="counter",
            service="task_processor",
            metadata={
                "task_id": task_id,
                "error": error,
                "priority": "high"
            },
            source="task_stream_processor"
        )


class AgentStreamProcessor(StreamProcessor):
    """Stream processor for agent-related events."""
    
    def __init__(self, config=None):
        super().__init__(
            name="agent_processor",
            input_topics=["agents"],
            output_topics=["agent_monitoring", "analytics", "alerts"],
            consumer_group="agent_stream_processor",
            config=config
        )
        
        # Agent monitoring
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.error_counts: Dict[str, int] = {}
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[BaseEvent]:
        """Process agent events."""
        try:
            event_type = EventType(event_data.get('event_type'))
            agent_id = event_data.get('agent_id', 'unknown')
            
            if event_type == EventType.AGENT_TASK_STARTED:
                return await self._handle_agent_task_started(event_data)
            elif event_type == EventType.AGENT_TASK_COMPLETED:
                return await self._handle_agent_task_completed(event_data)
            elif event_type == EventType.AGENT_ERROR:
                return await self._handle_agent_error(event_data)
            else:
                logger.debug(f"Unhandled agent event type: {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing agent event: {e}")
            return None
    
    async def _handle_agent_task_started(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle agent task start events."""
        agent_id = event_data.get('agent_id')
        task_id = event_data.get('task_id')
        
        # Initialize agent stats if needed
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                'tasks_started': 0,
                'tasks_completed': 0,
                'errors': 0,
                'last_active': None
            }
        
        self.agent_stats[agent_id]['tasks_started'] += 1
        self.agent_stats[agent_id]['last_active'] = datetime.now()
        
        return EventFactory.create_analytics_event(
            metric_name="agent_task_started",
            metric_value=1,
            metric_type="counter",
            service="agent_processor",
            metadata={
                "agent_id": agent_id,
                "task_id": task_id
            },
            source="agent_stream_processor"
        )
    
    async def _handle_agent_task_completed(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle agent task completion events."""
        agent_id = event_data.get('agent_id')
        task_id = event_data.get('task_id')
        result = event_data.get('result', {})
        
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id]['tasks_completed'] += 1
            self.agent_stats[agent_id]['last_active'] = datetime.now()
        
        return EventFactory.create_analytics_event(
            metric_name="agent_task_completed",
            metric_value=1,
            metric_type="counter",
            service="agent_processor",
            metadata={
                "agent_id": agent_id,
                "task_id": task_id,
                "success": result.get('success', True)
            },
            source="agent_stream_processor"
        )
    
    async def _handle_agent_error(self, event_data: Dict[str, Any]) -> BaseEvent:
        """Handle agent error events."""
        agent_id = event_data.get('agent_id')
        error_type = event_data.get('error_type', 'unknown')
        error_message = event_data.get('error_message', '')
        
        # Update error tracking
        if agent_id not in self.error_counts:
            self.error_counts[agent_id] = 0
        self.error_counts[agent_id] += 1
        
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id]['errors'] += 1
        
        # Create high-priority alert if error rate is high
        priority = "high" if self.error_counts[agent_id] > 5 else "medium"
        
        return EventFactory.create_analytics_event(
            metric_name="agent_error",
            metric_value=1,
            metric_type="counter",
            service="agent_processor",
            metadata={
                "agent_id": agent_id,
                "error_type": error_type,
                "error_message": error_message,
                "error_count": self.error_counts[agent_id],
                "priority": priority
            },
            source="agent_stream_processor"
        )


class AnalyticsStreamProcessor(StreamProcessor):
    """Stream processor for analytics aggregation."""
    
    def __init__(self, config=None):
        super().__init__(
            name="analytics_processor",
            input_topics=["analytics"],
            output_topics=["metrics", "dashboards", "alerts"],
            consumer_group="analytics_stream_processor",
            config=config
        )
        
        # Analytics aggregation
        self.metric_aggregations: Dict[str, Dict[str, Any]] = {}
        self.time_windows = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[BaseEvent]:
        """Process analytics events."""
        try:
            metric_name = event_data.get('metric_name')
            metric_value = event_data.get('metric_value', 0)
            metric_type = event_data.get('metric_type', 'gauge')
            service = event_data.get('service', 'unknown')
            
            if not metric_name:
                return None
            
            # Aggregate metrics
            await self._aggregate_metric(
                metric_name, metric_value, metric_type, service, event_data
            )
            
            # Check for threshold alerts
            alert_event = await self._check_thresholds(
                metric_name, metric_value, service, event_data
            )
            
            return alert_event
            
        except Exception as e:
            logger.error(f"Error processing analytics event: {e}")
            return None
    
    async def _aggregate_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_type: str,
        service: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Aggregate metric data across time windows."""
        timestamp = datetime.now()
        
        # Create metric key
        metric_key = f"{service}.{metric_name}"
        
        if metric_key not in self.metric_aggregations:
            self.metric_aggregations[metric_key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0,
                'last_value': 0.0,
                'last_update': timestamp,
                'metric_type': metric_type
            }
        
        # Update aggregation
        agg = self.metric_aggregations[metric_key]
        agg['count'] += 1
        agg['sum'] += metric_value
        agg['min'] = min(agg['min'], metric_value)
        agg['max'] = max(agg['max'], metric_value)
        agg['avg'] = agg['sum'] / agg['count']
        agg['last_value'] = metric_value
        agg['last_update'] = timestamp
    
    async def _check_thresholds(
        self,
        metric_name: str,
        metric_value: float,
        service: str,
        event_data: Dict[str, Any]
    ) -> Optional[BaseEvent]:
        """Check if metrics exceed predefined thresholds."""
        # Define some basic thresholds
        thresholds = {
            'error_rate': {'high': 10, 'critical': 20},
            'response_time': {'high': 5.0, 'critical': 10.0},
            'agent_error': {'high': 5, 'critical': 10},
            'task_failed': {'high': 3, 'critical': 5}
        }
        
        # Check if this metric has thresholds
        for threshold_metric, levels in thresholds.items():
            if threshold_metric in metric_name:
                priority = None
                
                if metric_value >= levels['critical']:
                    priority = 'critical'
                elif metric_value >= levels['high']:
                    priority = 'high'
                
                if priority:
                    # Create alert event
                    return EventFactory.create_system_health_event(
                        component=service,
                        status="warning",
                        message=f"Threshold exceeded for {metric_name}: {metric_value}",
                        metadata={
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "threshold_type": priority,
                            "alert_type": "threshold_exceeded"
                        },
                        source="analytics_stream_processor"
                    )
        
        return None
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get current metric aggregation summary."""
        summary = {}
        
        for metric_key, agg in self.metric_aggregations.items():
            summary[metric_key] = {
                'count': agg['count'],
                'avg': agg['avg'],
                'min': agg['min'],
                'max': agg['max'],
                'last_value': agg['last_value'],
                'last_update': agg['last_update'].isoformat(),
                'metric_type': agg['metric_type']
            }
        
        return summary


class NotificationStreamProcessor(StreamProcessor):
    """Stream processor for notifications."""
    
    def __init__(self, config=None):
        super().__init__(
            name="notification_processor",
            input_topics=["notifications"],
            output_topics=["email", "slack", "webhooks"],
            consumer_group="notification_stream_processor",
            config=config
        )
        
        # Notification tracking
        self.notification_stats = {
            'total_notifications': 0,
            'by_type': {},
            'by_priority': {}
        }
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[BaseEvent]:
        """Process notification events."""
        try:
            notification_type = event_data.get('notification_type', 'info')
            priority = event_data.get('priority', 'medium')
            recipient = event_data.get('recipient')
            message_text = event_data.get('message', '')
            
            # Update stats
            self.notification_stats['total_notifications'] += 1
            self.notification_stats['by_type'][notification_type] = (
                self.notification_stats['by_type'].get(notification_type, 0) + 1
            )
            self.notification_stats['by_priority'][priority] = (
                self.notification_stats['by_priority'].get(priority, 0) + 1
            )
            
            # Route notification based on priority and type
            if priority in ['high', 'critical']:
                # Send to immediate notification channels
                logger.info(f"High-priority notification: {message_text}")
            
            # Create processing confirmation event
            return EventFactory.create_analytics_event(
                metric_name="notification_processed",
                metric_value=1,
                metric_type="counter",
                service="notification_processor",
                metadata={
                    "notification_type": notification_type,
                    "priority": priority,
                    "recipient": recipient
                },
                source="notification_stream_processor"
            )
            
        except Exception as e:
            logger.error(f"Error processing notification event: {e}")
            return None