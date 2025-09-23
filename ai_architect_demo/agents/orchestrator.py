"""Agent orchestrator for coordinating multi-agent AI system.

This module provides the central coordination system that manages agent
lifecycles, task delegation, load balancing, and inter-agent communication.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Any
from collections import defaultdict

from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.base_agent import (
    BaseAgent, AgentTask, AgentMessage, AgentRole, AgentStatus, TaskPriority
)

logger = get_logger(__name__)


class TaskQueue:
    """Priority-based task queue for the orchestrator."""
    
    def __init__(self):
        self.queues: Dict[TaskPriority, List[AgentTask]] = {
            priority: [] for priority in TaskPriority
        }
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task: AgentTask) -> None:
        """Add a task to the appropriate priority queue.
        
        Args:
            task: Task to enqueue
        """
        async with self._lock:
            self.queues[task.priority].append(task)
    
    async def dequeue(self) -> Optional[AgentTask]:
        """Get the next highest priority task.
        
        Returns:
            Next task or None if queues are empty
        """
        async with self._lock:
            # Check queues in priority order (highest first)
            for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
                if self.queues[priority]:
                    return self.queues[priority].pop(0)
            return None
    
    def size(self) -> int:
        """Get total number of tasks in all queues."""
        return sum(len(queue) for queue in self.queues.values())
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get size of each priority queue."""
        return {
            priority.name.lower(): len(queue)
            for priority, queue in self.queues.items()
        }


class AgentOrchestrator:
    """Central orchestrator for managing AI agents and task distribution."""
    
    def __init__(self, max_retry_attempts: int = 3):
        """Initialize the orchestrator.
        
        Args:
            max_retry_attempts: Maximum task retry attempts
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[AgentRole, List[str]] = defaultdict(list)
        self.task_queue = TaskQueue()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.failed_tasks: List[AgentTask] = []
        self.max_retry_attempts = max_retry_attempts
        
        # Message routing
        self.message_handlers: Dict[str, callable] = {}
        
        # Performance tracking
        self.orchestrator_start_time = datetime.now()
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        
        # Task routing strategy
        self.routing_strategies = {
            "round_robin": self._round_robin_routing,
            "load_based": self._load_based_routing,
            "capability_based": self._capability_based_routing
        }
        self.current_routing_strategy = "capability_based"
        
        # Background task for processing queue
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("Agent orchestrator initialized")
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            agent: Agent to register
        """
        log_function_call("register_agent", agent_id=agent.agent_id, role=agent.role.value)
        
        self.agents[agent.agent_id] = agent
        self.agent_types[agent.role].append(agent.agent_id)
        
        logger.info(f"Registered agent {agent.name} ({agent.agent_id}) with role {agent.role.value}")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Clean up agent
            await agent.cleanup()
            
            # Remove from tracking
            self.agent_types[agent.role].remove(agent_id)
            del self.agents[agent_id]
            
            logger.info(f"Unregistered agent {agent_id}")
        else:
            logger.warning(f"Attempted to unregister unknown agent {agent_id}")
    
    async def submit_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        requester_id: Optional[str] = None,
        preferred_agent_role: Optional[AgentRole] = None
    ) -> str:
        """Submit a task for processing.
        
        Args:
            task_type: Type of task to process
            data: Task data
            priority: Task priority
            requester_id: ID of the requester
            preferred_agent_role: Preferred agent role for task
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            requester_id=requester_id
        )
        
        # Add preferred role to task data if specified
        if preferred_agent_role:
            task.data["preferred_agent_role"] = preferred_agent_role.value
        
        await self.task_queue.enqueue(task)
        
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a specific task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status information
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "active",
                "assigned_agent": getattr(task, 'assigned_agent_id', None),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "retry_count": task.retry_count
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task.result,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "processing_time": (
                        (task.completed_at - task.started_at).total_seconds()
                        if task.started_at and task.completed_at else None
                    )
                }
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": task.error,
                    "retry_count": task.retry_count,
                    "failed_at": task.completed_at.isoformat() if task.completed_at else None
                }
        
        # Check if task is still in queue
        # Note: This is a simplified check - in production, you'd want a more efficient lookup
        return {
            "task_id": task_id,
            "status": "queued",
            "queue_position": "unknown"  # Would need to implement queue position tracking
        }
    
    async def start_processing(self) -> None:
        """Start the background task processing loop."""
        if self._processing_task and not self._processing_task.done():
            logger.warning("Processing already started")
            return
        
        self._shutdown_event.clear()
        self._processing_task = asyncio.create_task(self._process_tasks())
        
        logger.info("Started task processing")
    
    async def stop_processing(self) -> None:
        """Stop the background task processing loop."""
        self._shutdown_event.set()
        
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Task processing shutdown timeout")
                self._processing_task.cancel()
        
        logger.info("Stopped task processing")
    
    async def _process_tasks(self) -> None:
        """Main task processing loop."""
        logger.info("Task processing loop started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                task = await self.task_queue.dequeue()
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1.0)
                    continue
                
                # Find suitable agent for the task
                agent = await self._select_agent_for_task(task)
                
                if agent is None:
                    # No suitable agent available, re-queue the task
                    await asyncio.sleep(5.0)  # Wait before re-queueing
                    await self.task_queue.enqueue(task)
                    continue
                
                # Assign task to agent
                await self._assign_task_to_agent(task, agent)
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("Task processing loop stopped")
    
    async def _select_agent_for_task(self, task: AgentTask) -> Optional[BaseAgent]:
        """Select the best agent for a task based on current routing strategy.
        
        Args:
            task: Task to assign
            
        Returns:
            Selected agent or None if no suitable agent available
        """
        strategy_func = self.routing_strategies.get(self.current_routing_strategy)
        if strategy_func:
            return await strategy_func(task)
        else:
            logger.error(f"Unknown routing strategy: {self.current_routing_strategy}")
            return await self._capability_based_routing(task)
    
    async def _capability_based_routing(self, task: AgentTask) -> Optional[BaseAgent]:
        """Select agent based on capabilities and current load.
        
        Args:
            task: Task to assign
            
        Returns:
            Best available agent or None
        """
        suitable_agents = []
        
        # Check if task specifies preferred agent role
        preferred_role = task.data.get("preferred_agent_role")
        if preferred_role:
            try:
                role_enum = AgentRole(preferred_role)
                candidate_ids = self.agent_types.get(role_enum, [])
            except ValueError:
                candidate_ids = []
        else:
            # Check all agents
            candidate_ids = list(self.agents.keys())
        
        # Filter agents by capability and availability
        for agent_id in candidate_ids:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            # Check if agent supports this task type
            if task.task_type not in agent.capabilities.supported_tasks:
                continue
            
            # Check if agent has capacity
            if len(agent.current_tasks) >= agent.capabilities.max_concurrent_tasks:
                continue
            
            # Agent is suitable
            suitable_agents.append((agent, len(agent.current_tasks)))
        
        if not suitable_agents:
            return None
        
        # Select agent with lowest current load
        suitable_agents.sort(key=lambda x: x[1])
        return suitable_agents[0][0]
    
    async def _load_based_routing(self, task: AgentTask) -> Optional[BaseAgent]:
        """Select agent based purely on current load.
        
        Args:
            task: Task to assign
            
        Returns:
            Agent with lowest load or None
        """
        available_agents = [
            (agent, len(agent.current_tasks))
            for agent in self.agents.values()
            if (len(agent.current_tasks) < agent.capabilities.max_concurrent_tasks and
                task.task_type in agent.capabilities.supported_tasks)
        ]
        
        if not available_agents:
            return None
        
        # Return agent with lowest load
        available_agents.sort(key=lambda x: x[1])
        return available_agents[0][0]
    
    async def _round_robin_routing(self, task: AgentTask) -> Optional[BaseAgent]:
        """Simple round-robin agent selection.
        
        Args:
            task: Task to assign
            
        Returns:
            Next agent in rotation or None
        """
        # This is a simplified implementation
        # In production, you'd want to maintain proper round-robin state
        return await self._load_based_routing(task)
    
    async def _assign_task_to_agent(self, task: AgentTask, agent: BaseAgent) -> None:
        """Assign a task to a specific agent.
        
        Args:
            task: Task to assign
            agent: Agent to receive the task
        """
        try:
            # Track the assignment
            self.active_tasks[task.task_id] = task
            task.assigned_agent_id = agent.agent_id
            
            # Add task to agent
            await agent.add_task(task)
            
            # Monitor task completion
            asyncio.create_task(self._monitor_task(task, agent))
            
            logger.info(f"Assigned task {task.task_id} to agent {agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to assign task {task.task_id} to agent {agent.agent_id}: {e}")
            
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
            
            # Re-queue the task if it hasn't exceeded retry limit
            if task.retry_count < self.max_retry_attempts:
                task.retry_count += 1
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self.task_queue.enqueue(task)
            else:
                task.error = f"Failed to assign after {self.max_retry_attempts} attempts: {e}"
                self.failed_tasks.append(task)
    
    async def _monitor_task(self, task: AgentTask, agent: BaseAgent) -> None:
        """Monitor task completion and handle results.
        
        Args:
            task: Task to monitor
            agent: Agent processing the task
        """
        # Wait for task completion (check periodically)
        while task.task_id in agent.current_tasks:
            await asyncio.sleep(1.0)
        
        # Task is complete, process the result
        await self._handle_task_completion(task)
    
    async def _handle_task_completion(self, task: AgentTask) -> None:
        """Handle completed task results and cleanup.
        
        Args:
            task: Completed task
        """
        # Remove from active tasks
        self.active_tasks.pop(task.task_id, None)
        
        if task.error:
            # Task failed
            if task.retry_count >= self.max_retry_attempts:
                self.failed_tasks.append(task)
                logger.error(f"Task {task.task_id} failed permanently: {task.error}")
            else:
                # Retry the task
                task.retry_count += 1
                task.started_at = None
                task.completed_at = None
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self.task_queue.enqueue(task)
                logger.info(f"Re-queuing failed task {task.task_id} (attempt {task.retry_count})")
        else:
            # Task completed successfully
            self.completed_tasks.append(task)
            self.total_tasks_processed += 1
            
            if task.started_at and task.completed_at:
                processing_time = (task.completed_at - task.started_at).total_seconds()
                self.total_processing_time += processing_time
            
            logger.info(f"Task {task.task_id} completed successfully")
    
    async def route_message(self, message: AgentMessage) -> None:
        """Route a message between agents.
        
        Args:
            message: Message to route
        """
        receiver = self.agents.get(message.receiver_id)
        if receiver:
            await receiver.receive_message(message)
        else:
            logger.warning(f"Message routing failed: agent {message.receiver_id} not found")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status and metrics.
        
        Returns:
            Status dictionary
        """
        uptime = (datetime.now() - self.orchestrator_start_time).total_seconds()
        
        avg_processing_time = (
            self.total_processing_time / self.total_tasks_processed
            if self.total_tasks_processed > 0 else 0
        )
        
        return {
            "orchestrator_uptime": uptime,
            "total_agents": len(self.agents),
            "agent_roles": {
                role.value: len(agents) for role, agents in self.agent_types.items()
            },
            "queue_sizes": self.task_queue.get_queue_sizes(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks_processed": self.total_tasks_processed,
            "average_processing_time": avg_processing_time,
            "routing_strategy": self.current_routing_strategy,
            "agent_statuses": {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator and all agents."""
        logger.info("Cleaning up orchestrator")
        
        # Stop processing
        await self.stop_processing()
        
        # Cleanup all agents
        cleanup_tasks = [agent.cleanup() for agent in self.agents.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.agents.clear()
        self.agent_types.clear()
        
        logger.info("Orchestrator cleanup complete")