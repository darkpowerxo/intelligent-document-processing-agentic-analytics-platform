"""Multi-agent AI system for enterprise-grade document processing and business intelligence.

This package provides a comprehensive agent orchestration system with specialized
agents for different types of AI tasks, coordinated through a central orchestrator.
"""

from ai_architect_demo.agents.orchestrator import AgentOrchestrator, TaskQueue
from ai_architect_demo.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentStatus,
    TaskPriority,
    AgentTask,
    AgentMessage,
    AgentCapabilities
)
from ai_architect_demo.agents.document_analyzer import DocumentAnalyzerAgent
from ai_architect_demo.agents.business_intelligence import BusinessIntelligenceAgent
from ai_architect_demo.agents.quality_assurance import (
    QualityAssuranceAgent,
    QualityLevel,
    QualityMetric,
    QualityStandard
)

__all__ = [
    # Core orchestration
    "AgentOrchestrator",
    "TaskQueue",
    
    # Base classes and enums
    "BaseAgent",
    "AgentRole",
    "AgentStatus", 
    "TaskPriority",
    "AgentTask",
    "AgentMessage",
    "AgentCapabilities",
    
    # Specialized agents
    "DocumentAnalyzerAgent",
    "BusinessIntelligenceAgent", 
    "QualityAssuranceAgent",
    
    # Quality assurance classes
    "QualityLevel",
    "QualityMetric", 
    "QualityStandard"
]