"""Integration layer for connecting the streaming system with the agentic AI system.

This module provides seamless integration between the real-time streaming
architecture and the existing agent orchestrator and specialized agents.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.orchestrator import AgentOrchestrator
from ai_architect_demo.agents.document_analyzer import DocumentAnalyzerAgent
from ai_architect_demo.agents.business_intelligence import BusinessIntelligenceAgent
from ai_architect_demo.agents.quality_assurance import QualityAssuranceAgent
from ai_architect_demo.agents.base_agent import AgentTask, TaskPriority
from ai_architect_demo.streaming.event_schemas import EventType, EventFactory
from ai_architect_demo.streaming.kafka_client import KafkaProducer, KafkaConsumer
from ai_architect_demo.streaming.stream_processors import StreamProcessor

logger = get_logger(__name__)


class AgentStreamingIntegration:
    """Integration layer between agents and streaming system."""
    
    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize the integration layer.
        
        Args:
            orchestrator: The agent orchestrator to integrate with
        """
        self.orchestrator = orchestrator
        self.kafka_producer = KafkaProducer()
        self.kafka_consumer = KafkaConsumer(
            topics=["agent_tasks", "document_processing", "analysis_requests"],
            group_id="agent_streaming_integration"
        )
        
        # Track agent activities
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
    
    async def start(self) -> None:
        """Start the streaming integration."""
        log_function_call("start", integration="agent_streaming")
        
        # Connect Kafka clients
        await self.kafka_producer.connect()
        await self.kafka_consumer.connect()
        
        # Setup event handlers
        self.kafka_consumer.add_handler(EventType.TASK_CREATED, self._handle_task_created)
        self.kafka_consumer.add_handler(EventType.DOCUMENT_UPLOADED, self._handle_document_uploaded)
        self.kafka_consumer.add_handler(EventType.AGENT_STATUS_CHANGED, self._handle_agent_task_monitoring)
        
        # Hook into orchestrator for real-time notifications
        self._setup_orchestrator_hooks()
        
        self.is_running = True
        logger.info("Agent streaming integration started")
        
        # Start consuming
        await self.kafka_consumer.start_consuming()
    
    async def stop(self) -> None:
        """Stop the streaming integration."""
        log_function_call("stop", integration="agent_streaming")
        
        self.is_running = False
        
        await self.kafka_consumer.disconnect()
        await self.kafka_producer.disconnect()
        
        logger.info("Agent streaming integration stopped")
    
    def _setup_orchestrator_hooks(self) -> None:
        """Setup hooks into the orchestrator for real-time event streaming."""
        
        # Override orchestrator's task execution to emit events
        original_execute_task = self.orchestrator.execute_task
        
        async def execute_task_with_streaming(*args, **kwargs):
            task_id = kwargs.get('task_id', 'unknown')
            
            # Emit task start event
            await self._emit_agent_task_started(task_id, kwargs.get('agent_id', 'unknown'))
            
            try:
                # Execute original task
                result = await original_execute_task(*args, **kwargs)
                
                # Emit task completion event
                await self._emit_agent_task_completed(task_id, result)
                
                return result
                
            except Exception as e:
                # Emit task error event
                await self._emit_agent_error(task_id, str(e))
                raise
        
        self.orchestrator.execute_task = execute_task_with_streaming
    
    async def _handle_task_created(self, event_data: Dict[str, Any], message) -> None:
        """Handle task creation events from the streaming system."""
        try:
            task_id = event_data.get('task_id')
            task_type = event_data.get('task_type', 'general')
            parameters = event_data.get('parameters', {})
            
            logger.info(f"Processing streaming task: {task_id} ({task_type})")
            
            # Route task to appropriate agent
            if task_type == 'document_analysis':
                await self._route_to_document_analyzer(task_id, parameters)
            elif task_type == 'business_intelligence':
                await self._route_to_business_intelligence(task_id, parameters)
            elif task_type == 'quality_assurance':
                await self._route_to_quality_assurance(task_id, parameters)
            else:
                # Use orchestrator for general tasks
                await self._route_to_orchestrator(task_id, task_type, parameters)
            
        except Exception as e:
            logger.error(f"Error handling task creation: {e}")
    
    async def _handle_document_uploaded(self, event_data: Dict[str, Any], message) -> None:
        """Handle document upload events."""
        try:
            document_id = event_data.get('document_id')
            file_path = event_data.get('file_path')
            
            # Automatically trigger document analysis
            task_id = f"doc_analysis_{document_id}_{int(datetime.now().timestamp())}"
            
            await self._route_to_document_analyzer(task_id, {
                'document_id': document_id,
                'file_path': file_path,
                'auto_triggered': True
            })
            
        except Exception as e:
            logger.error(f"Error handling document upload: {e}")
    
    async def _handle_agent_task_monitoring(self, event_data: Dict[str, Any], message) -> None:
        """Handle agent task monitoring events."""
        task_id = event_data.get('task_id')
        agent_id = event_data.get('agent_id')
        
        # Track active task
        self.active_tasks[task_id] = {
            'agent_id': agent_id,
            'started_at': datetime.now(),
            'status': 'running'
        }
    
    async def _route_to_document_analyzer(self, task_id: str, parameters: Dict[str, Any]) -> None:
        """Route task to document analyzer agent."""
        try:
            # Create document analyzer if not exists
            analyzer = DocumentAnalyzerAgent()
            
            # Prepare analysis request
            document_path = parameters.get('file_path', parameters.get('document_path'))
            if not document_path:
                raise ValueError("No document path provided for analysis")
            
            # Create agent task
            task = AgentTask(
                task_id=task_id,
                task_type="document_analysis",
                priority=TaskPriority.MEDIUM,
                data={
                    "document_path": document_path,
                    "analysis_type": parameters.get('analysis_type', 'comprehensive'),
                    "context": parameters.get('context', {})
                }
            )
            
            # Execute analysis
            result = await analyzer.process_task(task)
            
            # Emit completion event
            completion_event = EventFactory.create_agent_event(
                agent_id="document_analyzer",
                action="task_completed",
                task_id=task_id,
                result=result,
                metadata={
                    "document_path": document_path,
                    "analysis_type": parameters.get('analysis_type', 'comprehensive')
                },
                source="agent_streaming_integration"
            )
            
            await self.kafka_producer.send_event("agents", completion_event)
            
        except Exception as e:
            await self._emit_agent_error(task_id, str(e), "document_analyzer")
    
    async def _route_to_business_intelligence(self, task_id: str, parameters: Dict[str, Any]) -> None:
        """Route task to business intelligence agent."""
        try:
            bi_agent = BusinessIntelligenceAgent()
            
            # Prepare BI request
            analysis_type = parameters.get('analysis_type', 'trend_analysis')
            data_source = parameters.get('data_source', {})
            
            # Create agent task
            task = AgentTask(
                task_id=task_id,
                task_type="business_intelligence",
                priority=TaskPriority.MEDIUM,
                data={
                    "analysis_type": analysis_type,
                    "data_source": data_source,
                    "time_period": parameters.get('time_period', '30d')
                }
            )
            
            # Execute BI analysis
            result = await bi_agent.process_task(task)
            
            # Emit completion event
            completion_event = EventFactory.create_agent_event(
                agent_id="business_intelligence",
                action="task_completed",
                task_id=task_id,
                result=result,
                metadata={
                    "analysis_type": analysis_type,
                    "data_source_type": type(data_source).__name__
                },
                source="agent_streaming_integration"
            )
            
            await self.kafka_producer.send_event("agents", completion_event)
            
        except Exception as e:
            await self._emit_agent_error(task_id, str(e), "business_intelligence")
    
    async def _route_to_quality_assurance(self, task_id: str, parameters: Dict[str, Any]) -> None:
        """Route task to quality assurance agent."""
        try:
            qa_agent = QualityAssuranceAgent()
            
            # Prepare QA request
            target_content = parameters.get('content', parameters.get('document_path'))
            if not target_content:
                raise ValueError("No content provided for quality assurance")
            
            # Create agent task
            task = AgentTask(
                task_id=task_id,
                task_type="quality_assurance",
                priority=TaskPriority.MEDIUM,
                data={
                    "content": target_content,
                    "review_type": parameters.get('review_type', 'comprehensive'),
                    "criteria": parameters.get('criteria', {})
                }
            )
            
            # Execute QA review
            result = await qa_agent.process_task(task)
            
            # Emit completion event
            completion_event = EventFactory.create_agent_event(
                agent_id="quality_assurance",
                action="task_completed",
                task_id=task_id,
                result=result,
                metadata={
                    "review_type": parameters.get('review_type', 'comprehensive'),
                    "content_type": type(target_content).__name__
                },
                source="agent_streaming_integration"
            )
            
            await self.kafka_producer.send_event("agents", completion_event)
            
        except Exception as e:
            await self._emit_agent_error(task_id, str(e), "quality_assurance")
    
    async def _route_to_orchestrator(self, task_id: str, task_type: str, parameters: Dict[str, Any]) -> None:
        """Route task to orchestrator for general processing."""
        try:
            # Use orchestrator's analyze_query method for general tasks
            query = parameters.get('query', parameters.get('description', f'Execute {task_type} task'))
            
            result = await self.orchestrator.analyze_query(
                query=query,
                context=parameters.get('context', {}),
                require_verification=parameters.get('require_verification', False)
            )
            
            # Emit completion event
            completion_event = EventFactory.create_agent_event(
                agent_id="orchestrator",
                action="task_completed",
                task_id=task_id,
                result=result,
                metadata={
                    "task_type": task_type,
                    "query": query
                },
                source="agent_streaming_integration"
            )
            
            await self.kafka_producer.send_event("agents", completion_event)
            
        except Exception as e:
            await self._emit_agent_error(task_id, str(e), "orchestrator")
    
    async def _emit_agent_task_started(self, task_id: str, agent_id: str) -> None:
        """Emit agent task started event."""
        event = EventFactory.create_agent_event(
            agent_id=agent_id,
            action="task_started",
            task_id=task_id,
            metadata={"started_at": datetime.now().isoformat()},
            source="agent_streaming_integration"
        )
        
        await self.kafka_producer.send_event("agents", event)
    
    async def _emit_agent_task_completed(self, task_id: str, result: Any) -> None:
        """Emit agent task completed event."""
        event = EventFactory.create_agent_event(
            agent_id=self.active_tasks.get(task_id, {}).get('agent_id', 'unknown'),
            action="task_completed",
            task_id=task_id,
            result=result,
            metadata={"completed_at": datetime.now().isoformat()},
            source="agent_streaming_integration"
        )
        
        await self.kafka_producer.send_event("agents", event)
        
        # Remove from active tasks
        self.active_tasks.pop(task_id, None)
    
    async def _emit_agent_error(self, task_id: str, error_message: str, agent_id: str = "unknown") -> None:
        """Emit agent error event."""
        event = EventFactory.create_agent_event(
            agent_id=agent_id,
            action="error",
            task_id=task_id,
            error_message=error_message,
            metadata={
                "error_at": datetime.now().isoformat(),
                "error_type": "task_execution_error"
            },
            source="agent_streaming_integration"
        )
        
        await self.kafka_producer.send_event("agents", event)
        
        # Remove from active tasks
        self.active_tasks.pop(task_id, None)


class StreamingAgentProcessor(StreamProcessor):
    """Stream processor specifically for agent integration."""
    
    def __init__(self, orchestrator: AgentOrchestrator, config=None):
        super().__init__(
            name="streaming_agent_processor",
            input_topics=["agent_tasks", "document_processing"],
            output_topics=["agents", "analytics", "notifications"],
            consumer_group="streaming_agent_processor",
            config=config
        )
        
        self.orchestrator = orchestrator
        self.integration = AgentStreamingIntegration(orchestrator)
    
    async def start(self) -> None:
        """Start the streaming agent processor."""
        # Start the integration layer
        asyncio.create_task(self.integration.start())
        
        # Start the base processor
        await super().start()
    
    async def stop(self) -> None:
        """Stop the streaming agent processor."""
        await self.integration.stop()
        await super().stop()
    
    async def process_message(self, event_data: Dict[str, Any], message) -> Optional[Union[Any, Dict[str, Any]]]:
        """Process messages by routing them to the integration layer."""
        # The integration layer handles the actual processing
        # This just creates analytics events for monitoring
        
        event_type = event_data.get('event_type')
        
        # Create analytics event for processing
        analytics_event = EventFactory.create_analytics_event(
            metric_name="agent_integration_processed",
            metric_value=1,
            metric_type="counter",
            service="streaming_agent_processor",
            metadata={
                "event_type": event_type,
                "topic": message.topic
            },
            source="streaming_agent_processor"
        )
        
        return analytics_event


class StreamingNotificationService:
    """Service for sending real-time notifications about agent activities."""
    
    def __init__(self):
        """Initialize the notification service."""
        self.kafka_producer = KafkaProducer()
        self.is_running = False
    
    async def start(self) -> None:
        """Start the notification service."""
        await self.kafka_producer.connect()
        self.is_running = True
        logger.info("Streaming notification service started")
    
    async def stop(self) -> None:
        """Stop the notification service."""
        self.is_running = False
        await self.kafka_producer.disconnect()
        logger.info("Streaming notification service stopped")
    
    async def notify_task_completion(
        self,
        task_id: str,
        agent_id: str,
        result: Any,
        recipient: str = "system_admin"
    ) -> None:
        """Send notification about task completion."""
        notification_event = EventFactory.create_notification_event(
            notification_type="task_completion",
            recipient=recipient,
            message=f"Task {task_id} completed by {agent_id}",
            metadata={
                "task_id": task_id,
                "agent_id": agent_id,
                "result_summary": str(result)[:200]  # Truncate for brevity
            },
            source="streaming_notification_service"
        )
        
        await self.kafka_producer.send_event("notifications", notification_event)
    
    async def notify_agent_error(
        self,
        agent_id: str,
        error_message: str,
        task_id: str = None,
        recipient: str = "system_admin"
    ) -> None:
        """Send notification about agent error."""
        notification_event = EventFactory.create_notification_event(
            notification_type="agent_error",
            recipient=recipient,
            message=f"Agent {agent_id} encountered an error: {error_message}",
            priority="high",
            metadata={
                "agent_id": agent_id,
                "task_id": task_id,
                "error_message": error_message
            },
            source="streaming_notification_service"
        )
        
        await self.kafka_producer.send_event("notifications", notification_event)
    
    async def notify_system_event(
        self,
        event_type: str,
        message: str,
        priority: str = "medium",
        recipient: str = "system_admin"
    ) -> None:
        """Send general system notification."""
        notification_event = EventFactory.create_notification_event(
            notification_type=event_type,
            recipient=recipient,
            message=message,
            priority=priority,
            source="streaming_notification_service"
        )
        
        await self.kafka_producer.send_event("notifications", notification_event)


async def create_integrated_streaming_system(orchestrator: AgentOrchestrator) -> Dict[str, Any]:
    """Create a fully integrated streaming system with agent support.
    
    Args:
        orchestrator: The agent orchestrator to integrate
        
    Returns:
        Dictionary containing all streaming components
    """
    # Create integration layer
    agent_integration = AgentStreamingIntegration(orchestrator)
    
    # Create streaming agent processor
    agent_processor = StreamingAgentProcessor(orchestrator)
    
    # Create notification service
    notification_service = StreamingNotificationService()
    
    # Create monitoring components
    from ai_architect_demo.streaming.real_time_monitor import RealTimeMonitor
    monitor = RealTimeMonitor("integrated_stream_monitor")
    
    # Create event dispatcher with agent-specific routes
    from ai_architect_demo.streaming.event_dispatcher import EventDispatcher, PredefinedRoutes
    dispatcher = EventDispatcher("agent_integrated_dispatcher")
    
    # Add predefined routes
    routes = PredefinedRoutes.create_all_standard_routes()
    for route in routes:
        dispatcher.add_route(route)
    
    return {
        'agent_integration': agent_integration,
        'agent_processor': agent_processor,
        'notification_service': notification_service,
        'monitor': monitor,
        'dispatcher': dispatcher
    }