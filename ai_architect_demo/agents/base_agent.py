"""Base agent class for the agentic AI system.

This module provides the foundational Agent class that all specialized
agents inherit from, along with common functionality for LLM interaction,
task management, and communication protocols.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, Field

from ai_architect_demo.core.config import settings
from ai_architect_demo.core.logging import get_logger, log_function_call

logger = get_logger(__name__)


class AgentRole(Enum):
    """Agent roles in the system."""
    ORCHESTRATOR = "orchestrator"
    DOCUMENT_ANALYZER = "document_analyzer"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    QUALITY_ASSURANCE = "quality_assurance"
    DATA_PROCESSOR = "data_processor"
    MODEL_EVALUATOR = "model_evaluator"


class AgentStatus(Enum):
    """Agent status indicators."""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Task representation for agents."""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    requester_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class AgentMessage(BaseModel):
    """Message format for agent communication."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Agent capabilities and metadata."""
    supported_tasks: List[str]
    max_concurrent_tasks: int
    average_processing_time: Optional[float] = None
    success_rate: Optional[float] = None
    specializations: List[str] = Field(default_factory=list)
    required_resources: List[str] = Field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all AI agents in the system."""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        name: str,
        description: str,
        ollama_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1:latest"
    ):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent's role in the system
            name: Human-readable agent name
            description: Description of agent's purpose
            ollama_endpoint: Ollama server endpoint
            model_name: LLM model to use
        """
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.description = description
        self.ollama_endpoint = ollama_endpoint
        self.model_name = model_name
        
        # Initialize logger for this agent
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.current_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.message_queue: List[AgentMessage] = []
        
        # Performance tracking
        self.total_tasks_completed = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Agent capabilities
        self.capabilities = self._define_capabilities()
        
        # HTTP client for LLM communication
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        logger.info(f"Initialized agent {self.name} ({self.agent_id}) with role {self.role.value}")
    
    @abstractmethod
    def _define_capabilities(self) -> AgentCapabilities:
        """Define the agent's capabilities and supported tasks."""
        pass
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a specific task. Must be implemented by subclasses.
        
        Args:
            task: The task to process
            
        Returns:
            Task result dictionary
        """
        pass
    
    async def add_task(self, task: AgentTask) -> None:
        """Add a task to the agent's queue.
        
        Args:
            task: Task to add
        """
        log_function_call("add_task", agent_id=self.agent_id, task_id=task.task_id)
        
        if len(self.current_tasks) >= self.capabilities.max_concurrent_tasks:
            raise ValueError(f"Agent {self.agent_id} is at maximum capacity")
        
        self.current_tasks[task.task_id] = task
        logger.info(f"Agent {self.agent_id} received task {task.task_id}")
        
        # Start processing in background
        asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: AgentTask) -> None:
        """Execute a task with error handling and retry logic.
        
        Args:
            task: Task to execute
        """
        task.started_at = datetime.now()
        self.status = AgentStatus.WORKING
        
        try:
            logger.info(f"Agent {self.agent_id} starting task {task.task_id}")
            
            # Process the task
            result = await self.process_task(task)
            
            # Update task with result
            task.result = result
            task.completed_at = datetime.now()
            
            # Update performance metrics
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self.total_tasks_completed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Agent {self.agent_id} completed task {task.task_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} task {task.task_id} failed: {e}")
            
            task.error = str(e)
            task.retry_count += 1
            self.error_count += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(task)
                return
            else:
                task.completed_at = datetime.now()
                logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
        
        finally:
            # Move task to completed and update status
            if task.task_id in self.current_tasks:
                self.completed_tasks.append(self.current_tasks.pop(task.task_id))
            
            self.status = AgentStatus.IDLE if not self.current_tasks else AgentStatus.WORKING
    
    async def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any]) -> None:
        """Send a message to another agent.
        
        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
        """
        message = AgentMessage(
            message_id=f"msg_{int(time.time() * 1000)}",
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        
        # In a full implementation, this would route through the orchestrator
        logger.info(f"Agent {self.agent_id} sending message to {receiver_id}: {message_type}")
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive and process a message from another agent.
        
        Args:
            message: Incoming message
        """
        self.message_queue.append(message)
        logger.info(f"Agent {self.agent_id} received message from {message.sender_id}: {message.message_type}")
        
        # Process message based on type
        await self._handle_message(message)
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming messages. Can be overridden by subclasses.
        
        Args:
            message: Message to handle
        """
        # Default message handling
        if message.message_type == "status_request":
            await self.send_message(
                message.sender_id,
                "status_response",
                {
                    "status": self.status.value,
                    "current_tasks": len(self.current_tasks),
                    "completed_tasks": self.total_tasks_completed
                }
            )
    
    async def query_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> str:
        """Query the local LLM via Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        log_function_call("query_llm", agent_id=self.agent_id, model=self.model_name)
        
        try:
            # Prepare the request
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens or -1
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Make the request to Ollama
            response = await self.http_client.post(
                f"{self.ollama_endpoint}/api/generate",
                json=request_data
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"LLM query failed for agent {self.agent_id}: {e}")
            raise
    
    async def query_llm_structured(
        self,
        prompt: str,
        response_format: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Query LLM for structured JSON response.
        
        Args:
            prompt: User prompt
            response_format: Expected JSON schema
            system_prompt: System prompt
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response
        """
        # Create a prompt that requests JSON format
        json_prompt = f"""
{prompt}

Please respond with valid JSON that matches this format:
{json.dumps(response_format, indent=2)}

Respond only with the JSON, no additional text.
"""
        
        # Add JSON format instruction to system prompt
        json_system_prompt = (system_prompt or "") + "\n\nAlways respond with valid JSON format."
        
        response = await self.query_llm(json_prompt, json_system_prompt, temperature)
        
        try:
            # Try to parse JSON response
            # Clean up the response (remove markdown formatting if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            return json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics.
        
        Returns:
            Status dictionary
        """
        avg_processing_time = (
            self.total_processing_time / self.total_tasks_completed
            if self.total_tasks_completed > 0 else 0
        )
        
        success_rate = (
            (self.total_tasks_completed - self.error_count) / self.total_tasks_completed
            if self.total_tasks_completed > 0 else 1.0
        )
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.total_tasks_completed,
            "error_count": self.error_count,
            "average_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "capabilities": self.capabilities.dict(),
            "message_queue_size": len(self.message_queue)
        }
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        await self.http_client.aclose()
        self.status = AgentStatus.OFFLINE
        logger.info(f"Agent {self.agent_id} cleaned up")