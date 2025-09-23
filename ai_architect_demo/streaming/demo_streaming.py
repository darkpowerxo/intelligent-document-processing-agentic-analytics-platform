"""Comprehensive demo system for the streaming architecture.

This module provides a complete demonstration of the streaming system
integrated with the agentic AI system, showing real-time event processing,
monitoring, and analytics capabilities.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.orchestrator import AgentOrchestrator
from ai_architect_demo.streaming.event_schemas import EventFactory, EventType
from ai_architect_demo.streaming.kafka_client import KafkaProducer
from ai_architect_demo.streaming.agent_integration import create_integrated_streaming_system
from ai_architect_demo.streaming.event_dispatcher import EventDispatcher, PredefinedRoutes
from ai_architect_demo.streaming.stream_processors import (
    DocumentStreamProcessor, TaskStreamProcessor, 
    AgentStreamProcessor, AnalyticsStreamProcessor
)
from ai_architect_demo.streaming.real_time_monitor import RealTimeMonitor

logger = get_logger(__name__)


class StreamingDemoConfig(BaseModel):
    """Configuration for the streaming demo."""
    
    demo_duration_minutes: int = 5
    event_generation_rate: float = 2.0  # events per second
    enable_monitoring: bool = True
    enable_alerts: bool = True
    simulate_errors: bool = True
    error_rate_percent: float = 5.0
    
    # Event type distribution (percentages)
    document_events_percent: float = 30.0
    task_events_percent: float = 25.0
    agent_events_percent: float = 20.0
    analytics_events_percent: float = 15.0
    notification_events_percent: float = 10.0


class StreamingEventSimulator:
    """Simulates realistic streaming events for demonstration."""
    
    def __init__(self, kafka_producer: KafkaProducer, config: StreamingDemoConfig):
        """Initialize the event simulator.
        
        Args:
            kafka_producer: Kafka producer for sending events
            config: Demo configuration
        """
        self.kafka_producer = kafka_producer
        self.config = config
        self.is_running = False
        self.events_generated = 0
        
        # Sample data for realistic event generation
        self.sample_documents = [
            {"id": "doc_001", "name": "quarterly_report.pdf", "size": 1024000},
            {"id": "doc_002", "name": "project_proposal.docx", "size": 512000},
            {"id": "doc_003", "name": "meeting_minutes.txt", "size": 8192},
            {"id": "doc_004", "name": "technical_spec.md", "size": 65536},
            {"id": "doc_005", "name": "user_manual.pdf", "size": 2048000}
        ]
        
        self.sample_agents = [
            "document_analyzer", "business_intelligence", "quality_assurance", 
            "content_reviewer", "data_processor", "report_generator"
        ]
        
        self.sample_tasks = [
            "analyze_document", "generate_report", "review_quality",
            "extract_insights", "process_data", "create_summary"
        ]
    
    async def start_simulation(self) -> None:
        """Start generating simulated events."""
        log_function_call("start_simulation", duration=self.config.demo_duration_minutes)
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=self.config.demo_duration_minutes)
        
        logger.info(f"Starting event simulation for {self.config.demo_duration_minutes} minutes")
        
        while self.is_running and datetime.now() < end_time:
            try:
                # Generate event based on distribution
                event_type = self._select_event_type()
                event = await self._generate_event(event_type)
                
                if event:
                    # Send to appropriate topic
                    topic = self._get_topic_for_event(event_type)
                    success = await self.kafka_producer.send_event(topic, event)
                    
                    if success:
                        self.events_generated += 1
                        if self.events_generated % 100 == 0:
                            logger.info(f"Generated {self.events_generated} events")
                    
                    # Simulate errors occasionally
                    if self.config.simulate_errors and random.random() < (self.config.error_rate_percent / 100):
                        await self._generate_error_event()
                
                # Wait based on event generation rate
                await asyncio.sleep(1.0 / self.config.event_generation_rate)
                
            except Exception as e:
                logger.error(f"Error generating event: {e}")
                await asyncio.sleep(1)
        
        self.is_running = False
        logger.info(f"Event simulation completed. Generated {self.events_generated} events")
    
    def stop_simulation(self) -> None:
        """Stop the event simulation."""
        self.is_running = False
    
    def _select_event_type(self) -> EventType:
        """Select event type based on distribution."""
        rand_val = random.random() * 100
        
        if rand_val < self.config.document_events_percent:
            return random.choice([EventType.DOCUMENT_UPLOADED, EventType.DOCUMENT_PROCESSED])
        elif rand_val < self.config.document_events_percent + self.config.task_events_percent:
            return random.choice([EventType.TASK_CREATED, EventType.TASK_STARTED, EventType.TASK_COMPLETED])
        elif rand_val < self.config.document_events_percent + self.config.task_events_percent + self.config.agent_events_percent:
            return random.choice([EventType.AGENT_TASK_STARTED, EventType.AGENT_TASK_COMPLETED])
        elif rand_val < (self.config.document_events_percent + self.config.task_events_percent + 
                        self.config.agent_events_percent + self.config.analytics_events_percent):
            return EventType.ANALYTICS_METRIC_UPDATE
        else:
            return EventType.NOTIFICATION_EVENT
    
    async def _generate_event(self, event_type: EventType):
        """Generate a specific type of event."""
        if event_type == EventType.DOCUMENT_UPLOADED:
            return await self._generate_document_uploaded_event()
        elif event_type == EventType.DOCUMENT_PROCESSED:
            return await self._generate_document_processed_event()
        elif event_type == EventType.TASK_CREATED:
            return await self._generate_task_created_event()
        elif event_type == EventType.TASK_STARTED:
            return await self._generate_task_started_event()
        elif event_type == EventType.TASK_COMPLETED:
            return await self._generate_task_completed_event()
        elif event_type == EventType.AGENT_TASK_STARTED:
            return await self._generate_agent_task_started_event()
        elif event_type == EventType.AGENT_TASK_COMPLETED:
            return await self._generate_agent_task_completed_event()
        elif event_type == EventType.ANALYTICS_METRIC_UPDATE:
            return await self._generate_analytics_event()
        elif event_type == EventType.NOTIFICATION_EVENT:
            return await self._generate_notification_event()
        
        return None
    
    async def _generate_document_uploaded_event(self):
        """Generate a document upload event."""
        doc = random.choice(self.sample_documents)
        return EventFactory.create_document_event(
            document_id=doc["id"],
            action="uploaded",
            status="pending",
            file_path=f"/uploads/{doc['name']}",
            metadata={
                "file_size": doc["size"],
                "uploaded_by": f"user_{random.randint(1, 10)}",
                "mime_type": "application/pdf" if doc["name"].endswith(".pdf") else "text/plain"
            },
            source="streaming_demo"
        )
    
    async def _generate_document_processed_event(self):
        """Generate a document processed event."""
        doc = random.choice(self.sample_documents)
        return EventFactory.create_document_event(
            document_id=doc["id"],
            action="processed",
            status="completed",
            file_path=f"/processed/{doc['name']}",
            metadata={
                "processing_time_ms": random.randint(100, 5000),
                "pages_analyzed": random.randint(1, 20),
                "confidence_score": round(random.uniform(0.8, 1.0), 2)
            },
            source="streaming_demo"
        )
    
    async def _generate_task_created_event(self):
        """Generate a task created event."""
        task_type = random.choice(self.sample_tasks)
        return EventFactory.create_task_event(
            task_id=f"task_{int(time.time())}_{random.randint(1000, 9999)}",
            task_type=task_type,
            action="created",
            status="pending",
            metadata={
                "priority": random.choice(["low", "medium", "high"]),
                "estimated_duration": random.randint(30, 300),
                "created_by": f"user_{random.randint(1, 10)}"
            },
            source="streaming_demo"
        )
    
    async def _generate_task_started_event(self):
        """Generate a task started event."""
        return EventFactory.create_task_event(
            task_id=f"task_{int(time.time())}_{random.randint(1000, 9999)}",
            task_type=random.choice(self.sample_tasks),
            action="started",
            status="running",
            metadata={"started_by": random.choice(self.sample_agents)},
            source="streaming_demo"
        )
    
    async def _generate_task_completed_event(self):
        """Generate a task completed event."""
        return EventFactory.create_task_event(
            task_id=f"task_{int(time.time())}_{random.randint(1000, 9999)}",
            task_type=random.choice(self.sample_tasks),
            action="completed",
            status="completed",
            metadata={
                "completion_time_ms": random.randint(1000, 10000),
                "success": random.choice([True, True, True, False])  # 75% success rate
            },
            source="streaming_demo"
        )
    
    async def _generate_agent_task_started_event(self):
        """Generate an agent task started event."""
        return EventFactory.create_agent_event(
            agent_id=random.choice(self.sample_agents),
            action="task_started",
            task_id=f"agent_task_{int(time.time())}_{random.randint(1000, 9999)}",
            metadata={"resource_allocation": random.randint(10, 100)},
            source="streaming_demo"
        )
    
    async def _generate_agent_task_completed_event(self):
        """Generate an agent task completed event."""
        return EventFactory.create_agent_event(
            agent_id=random.choice(self.sample_agents),
            action="task_completed",
            task_id=f"agent_task_{int(time.time())}_{random.randint(1000, 9999)}",
            result={
                "status": "success",
                "processing_time": random.randint(500, 5000),
                "confidence": round(random.uniform(0.7, 1.0), 2)
            },
            source="streaming_demo"
        )
    
    async def _generate_analytics_event(self):
        """Generate an analytics event."""
        metrics = [
            ("cpu_usage", random.uniform(10, 90), "gauge"),
            ("memory_usage", random.uniform(20, 80), "gauge"),
            ("requests_per_second", random.randint(1, 100), "counter"),
            ("error_count", random.randint(0, 5), "counter"),
            ("response_time_ms", random.randint(50, 2000), "histogram")
        ]
        
        metric_name, metric_value, metric_type = random.choice(metrics)
        
        return EventFactory.create_analytics_event(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            service=random.choice(["document_processor", "task_manager", "agent_orchestrator"]),
            source="streaming_demo"
        )
    
    async def _generate_notification_event(self):
        """Generate a notification event."""
        notification_types = ["info", "warning", "alert", "system_update"]
        priorities = ["low", "medium", "high"]
        
        return EventFactory.create_notification_event(
            notification_type=random.choice(notification_types),
            recipient=f"user_{random.randint(1, 10)}",
            message=f"System notification: {random.choice(['Task completed', 'New document available', 'System maintenance', 'Performance update'])}",
            priority=random.choice(priorities),
            source="streaming_demo"
        )
    
    async def _generate_error_event(self):
        """Generate an error event."""
        error_event = EventFactory.create_agent_event(
            agent_id=random.choice(self.sample_agents),
            action="error",
            task_id=f"error_task_{int(time.time())}",
            error_message=random.choice([
                "Connection timeout",
                "Invalid input data",
                "Resource exhausted",
                "Processing failed",
                "Authentication error"
            ]),
            metadata={"error_code": random.randint(400, 599)},
            source="streaming_demo"
        )
        
        await self.kafka_producer.send_event("agents", error_event)
    
    def _get_topic_for_event(self, event_type: EventType) -> str:
        """Get the appropriate Kafka topic for an event type."""
        topic_mapping = {
            EventType.DOCUMENT_UPLOADED: "documents",
            EventType.DOCUMENT_PROCESSED: "documents",
            EventType.TASK_CREATED: "tasks",
            EventType.TASK_STARTED: "tasks",
            EventType.TASK_COMPLETED: "tasks",
            EventType.TASK_FAILED: "tasks",
            EventType.AGENT_TASK_STARTED: "agents",
            EventType.AGENT_TASK_COMPLETED: "agents",
            EventType.AGENT_ERROR: "agents",
            EventType.ANALYTICS_METRIC_UPDATE: "analytics",
            EventType.NOTIFICATION_EVENT: "notifications",
            EventType.QUALITY_CHECK: "quality",
            EventType.BUSINESS_INSIGHT: "business_intelligence",
            EventType.SYSTEM_HEALTH: "system_health"
        }
        
        return topic_mapping.get(event_type, "system_events")


class StreamingDemoOrchestrator:
    """Orchestrates the complete streaming demo."""
    
    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize the demo orchestrator.
        
        Args:
            orchestrator: Agent orchestrator for integration
        """
        self.orchestrator = orchestrator
        self.config = StreamingDemoConfig()
        self.components: Dict[str, Any] = {}
        self.demo_stats = {
            'start_time': None,
            'end_time': None,
            'events_generated': 0,
            'events_processed': 0,
            'alerts_triggered': 0,
            'errors_encountered': 0
        }
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete streaming architecture demo.
        
        Returns:
            Demo results and statistics
        """
        log_function_call("run_complete_demo", duration=self.config.demo_duration_minutes)
        
        try:
            # Setup phase
            logger.info("Setting up streaming demo components...")
            await self._setup_demo_components()
            
            # Start all components
            logger.info("Starting streaming components...")
            await self._start_components()
            
            # Wait for components to stabilize
            await asyncio.sleep(2)
            
            # Run demonstration
            logger.info("Running streaming demonstration...")
            self.demo_stats['start_time'] = datetime.now()
            
            await self._run_demonstration()
            
            self.demo_stats['end_time'] = datetime.now()
            
            # Collect final statistics
            logger.info("Collecting demo results...")
            demo_results = await self._collect_demo_results()
            
            return demo_results
            
        except Exception as e:
            logger.error(f"Error during streaming demo: {e}")
            raise
        
        finally:
            # Cleanup
            logger.info("Cleaning up demo components...")
            await self._cleanup_components()
    
    async def _setup_demo_components(self) -> None:
        """Setup all demo components."""
        # Create integrated streaming system
        self.components = await create_integrated_streaming_system(self.orchestrator)
        
        # Add stream processors
        self.components['document_processor'] = DocumentStreamProcessor()
        self.components['task_processor'] = TaskStreamProcessor()
        self.components['agent_processor'] = AgentStreamProcessor()
        self.components['analytics_processor'] = AnalyticsStreamProcessor()
        
        # Create event simulator
        kafka_producer = KafkaProducer()
        await kafka_producer.connect()
        self.components['event_simulator'] = StreamingEventSimulator(kafka_producer, self.config)
        self.components['demo_producer'] = kafka_producer
    
    async def _start_components(self) -> None:
        """Start all streaming components."""
        start_tasks = []
        
        # Start core streaming components
        for name, component in self.components.items():
            if hasattr(component, 'start') and name != 'event_simulator':
                start_tasks.append(asyncio.create_task(component.start()))
                logger.info(f"Starting {name}...")
        
        # Start components concurrently
        await asyncio.gather(*start_tasks, return_exceptions=True)
        
        logger.info("All streaming components started")
    
    async def _run_demonstration(self) -> None:
        """Run the main demonstration."""
        # Start event simulation
        simulator = self.components['event_simulator']
        simulation_task = asyncio.create_task(simulator.start_simulation())
        
        # Monitor demo progress
        monitor_task = asyncio.create_task(self._monitor_demo_progress())
        
        # Wait for simulation to complete
        await asyncio.gather(simulation_task, monitor_task, return_exceptions=True)
    
    async def _monitor_demo_progress(self) -> None:
        """Monitor demo progress and log statistics."""
        start_time = datetime.now()
        
        while datetime.now() - start_time < timedelta(minutes=self.config.demo_duration_minutes):
            await asyncio.sleep(30)  # Progress update every 30 seconds
            
            # Collect current stats
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            events_generated = self.components['event_simulator'].events_generated
            
            logger.info(f"Demo progress: {elapsed:.1f}/{self.config.demo_duration_minutes} minutes, "
                       f"{events_generated} events generated")
            
            # Check component health
            monitor = self.components.get('monitor')
            if monitor:
                dashboard_data = monitor.get_dashboard_data()
                active_alerts = dashboard_data.get('active_alerts', 0)
                
                if active_alerts > self.demo_stats['alerts_triggered']:
                    self.demo_stats['alerts_triggered'] = active_alerts
                    logger.info(f"New alerts triggered: {active_alerts} total")
    
    async def _collect_demo_results(self) -> Dict[str, Any]:
        """Collect comprehensive demo results."""
        results = {
            'demo_config': self.config.model_dump(),
            'demo_stats': self.demo_stats.copy(),
            'component_stats': {},
            'event_distribution': {},
            'performance_metrics': {},
            'alerts_summary': {},
            'streaming_topology': {}
        }
        
        # Collect component statistics
        for name, component in self.components.items():
            if hasattr(component, 'get_stats'):
                results['component_stats'][name] = component.get_stats()
            elif hasattr(component, 'get_dashboard_data'):
                results['component_stats'][name] = component.get_dashboard_data()
        
        # Update final demo stats
        simulator = self.components['event_simulator']
        results['demo_stats']['events_generated'] = simulator.events_generated
        
        # Calculate demo duration
        if self.demo_stats['start_time'] and self.demo_stats['end_time']:
            duration = (self.demo_stats['end_time'] - self.demo_stats['start_time']).total_seconds()
            results['demo_stats']['duration_seconds'] = duration
            results['demo_stats']['events_per_second'] = simulator.events_generated / max(duration, 1)
        
        # Generate summary
        results['summary'] = {
            'demo_successful': True,
            'total_events': simulator.events_generated,
            'components_active': len([c for c in self.components.values() if hasattr(c, 'is_running') and getattr(c, 'is_running', False)]),
            'alerts_triggered': self.demo_stats['alerts_triggered'],
            'demo_duration_minutes': self.config.demo_duration_minutes
        }
        
        return results
    
    async def _cleanup_components(self) -> None:
        """Clean up all demo components."""
        cleanup_tasks = []
        
        # Stop all components
        for name, component in self.components.items():
            if hasattr(component, 'stop'):
                cleanup_tasks.append(asyncio.create_task(component.stop()))
                logger.info(f"Stopping {name}...")
        
        # Stop components concurrently
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("All demo components stopped")
    
    def update_config(self, **kwargs) -> None:
        """Update demo configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"Demo configuration updated: {kwargs}")


async def run_streaming_demo() -> Dict[str, Any]:
    """Run the complete streaming architecture demonstration.
    
    Returns:
        Comprehensive demo results
    """
    try:
        # Create orchestrator
        from ai_architect_demo.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        
        # Create and run demo
        demo = StreamingDemoOrchestrator(orchestrator)
        
        # Customize demo configuration
        demo.update_config(
            demo_duration_minutes=3,
            event_generation_rate=5.0,
            simulate_errors=True,
            error_rate_percent=3.0
        )
        
        # Run complete demo
        results = await demo.run_complete_demo()
        
        # Log summary
        summary = results.get('summary', {})
        logger.info(f"Streaming demo completed successfully!")
        logger.info(f"Total events generated: {summary.get('total_events', 0)}")
        logger.info(f"Components active: {summary.get('components_active', 0)}")
        logger.info(f"Alerts triggered: {summary.get('alerts_triggered', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Streaming demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run demo if executed directly
    import asyncio
    
    async def main():
        results = await run_streaming_demo()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())