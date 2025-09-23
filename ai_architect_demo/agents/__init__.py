"""Agentic AI system for AI Architect Demo.

This package provides a multi-agent orchestration system with:
- Specialized AI agents for different tasks
- Agent coordination and communication
- Local LLM integration
- Task delegation and result aggregation
"""

from .orchestrator import AgentOrchestrator, agent_orchestrator
from .base_agent import BaseAgent, AgentRole, AgentStatus
from .document_analyzer import DocumentAnalyzerAgent
from .business_intelligence import BusinessIntelligenceAgent
from .quality_assurance import QualityAssuranceAgent

__all__ = [
    "AgentOrchestrator",
    "agent_orchestrator", 
    "BaseAgent",
    "AgentRole",
    "AgentStatus",
    "DocumentAnalyzerAgent",
    "BusinessIntelligenceAgent", 
    "QualityAssuranceAgent"
]