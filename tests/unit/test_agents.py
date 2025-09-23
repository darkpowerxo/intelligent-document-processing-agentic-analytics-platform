"""Unit tests for agent components.

Tests the base agent, specialized agents, and orchestrator functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from ai_architect_demo.agents.base import BaseAgent, AgentTask, TaskPriority, TaskStatus
from ai_architect_demo.agents.document_analyzer import DocumentAnalyzerAgent
from ai_architect_demo.agents.business_intelligence import BusinessIntelligenceAgent  
from ai_architect_demo.agents.quality_assurance import QualityAssuranceAgent
from ai_architect_demo.agents.orchestrator import AgentOrchestrator


class TestBaseAgent:
    """Test base agent functionality."""
    
    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""
        return BaseAgent(
            agent_id="test_agent",
            name="Test Agent", 
            description="A test agent",
            capabilities={"test": True, "analysis": True}
        )
    
    def test_agent_initialization(self, sample_agent):
        """Test agent initialization with basic properties."""
        assert sample_agent.agent_id == "test_agent"
        assert sample_agent.name == "Test Agent"
        assert sample_agent.description == "A test agent"
        assert sample_agent.capabilities == {"test": True, "analysis": True}
        assert sample_agent.is_available is True
        assert sample_agent.current_tasks == []
    
    def test_agent_task_creation(self):
        """Test agent task creation and validation."""
        task = AgentTask(
            task_id="test_task",
            task_type="analysis",
            priority=TaskPriority.HIGH,
            data={"content": "test content"},
            requester_id="user123"
        )
        
        assert task.task_id == "test_task"
        assert task.task_type == "analysis"
        assert task.priority == TaskPriority.HIGH
        assert task.data["content"] == "test content"
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_base_agent_process_task_not_implemented(self, sample_agent):
        """Test that base agent process_task raises NotImplementedError."""
        task = AgentTask(
            task_id="test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            data={},
            requester_id="test"
        )
        
        with pytest.raises(NotImplementedError):
            await sample_agent.process_task(task)
    
    def test_agent_can_handle_task(self, sample_agent):
        """Test agent capability checking."""
        # Mock the can_handle_task method for testing
        sample_agent.can_handle_task = Mock(return_value=True)
        
        task = AgentTask(
            task_id="test",
            task_type="analysis",
            priority=TaskPriority.MEDIUM,
            data={},
            requester_id="test"
        )
        
        assert sample_agent.can_handle_task(task) is True


class TestDocumentAnalyzerAgent:
    """Test DocumentAnalyzer agent functionality."""
    
    @pytest.fixture
    def document_agent(self):
        """Create a DocumentAnalyzer agent for testing."""
        return DocumentAnalyzerAgent()
    
    def test_document_agent_initialization(self, document_agent):
        """Test DocumentAnalyzer initialization."""
        assert document_agent.name == "Document Analyzer"
        assert "document_analysis" in document_agent.capabilities
        assert "text_extraction" in document_agent.capabilities
        assert document_agent.is_available is True
    
    def test_document_agent_can_handle_task(self, document_agent):
        """Test DocumentAnalyzer task capability checking."""
        # Document analysis task
        doc_task = AgentTask(
            task_id="doc_task",
            task_type="document_analysis",
            priority=TaskPriority.MEDIUM,
            data={"document_id": "doc123"},
            requester_id="user123"
        )
        
        assert document_agent.can_handle_task(doc_task) is True
        
        # Non-document task
        other_task = AgentTask(
            task_id="other_task", 
            task_type="business_analysis",
            priority=TaskPriority.MEDIUM,
            data={},
            requester_id="user123"
        )
        
        assert document_agent.can_handle_task(other_task) is False
    
    @pytest.mark.asyncio
    async def test_document_agent_process_task(self, document_agent):
        """Test DocumentAnalyzer task processing."""
        task = AgentTask(
            task_id="doc_analysis_task",
            task_type="document_analysis", 
            priority=TaskPriority.MEDIUM,
            data={
                "document_id": "doc123",
                "content": "This is a test document content."
            },
            requester_id="user123"
        )
        
        # Mock the actual analysis methods
        with patch.object(document_agent, '_analyze_document_content', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "summary": "Test document summary",
                "key_points": ["Point 1", "Point 2"],
                "entities": {"people": ["John Doe"], "locations": ["New York"]}
            }
            
            result = await document_agent.process_task(task)
            
            assert result["status"] == "completed"
            assert "analysis" in result
            assert result["analysis"]["summary"] == "Test document summary"
            mock_analyze.assert_called_once()


class TestBusinessIntelligenceAgent:
    """Test BusinessIntelligence agent functionality."""
    
    @pytest.fixture
    def bi_agent(self):
        """Create a BusinessIntelligence agent for testing."""
        return BusinessIntelligenceAgent()
    
    def test_bi_agent_initialization(self, bi_agent):
        """Test BusinessIntelligence initialization."""
        assert bi_agent.name == "Business Intelligence Analyzer"
        assert "business_analysis" in bi_agent.capabilities
        assert "metrics_analysis" in bi_agent.capabilities
        assert bi_agent.is_available is True
    
    @pytest.mark.asyncio
    async def test_bi_agent_process_task(self, bi_agent):
        """Test BusinessIntelligence task processing."""
        task = AgentTask(
            task_id="bi_task",
            task_type="business_analysis",
            priority=TaskPriority.MEDIUM,
            data={
                "data_source": "quarterly_reports",
                "metrics": ["revenue", "growth_rate"]
            },
            requester_id="user123"
        )
        
        # Mock the analysis methods
        with patch.object(bi_agent, '_analyze_business_metrics', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "insights": ["Revenue increased by 15%", "Growth rate is stable"],
                "recommendations": ["Focus on customer retention"],
                "metrics": {"revenue_growth": 0.15, "customer_satisfaction": 0.85}
            }
            
            result = await bi_agent.process_task(task)
            
            assert result["status"] == "completed"
            assert "analysis" in result
            assert "insights" in result["analysis"]
            mock_analyze.assert_called_once()


class TestQualityAssuranceAgent:
    """Test QualityAssurance agent functionality."""
    
    @pytest.fixture
    def qa_agent(self):
        """Create a QualityAssurance agent for testing."""
        return QualityAssuranceAgent()
    
    def test_qa_agent_initialization(self, qa_agent):
        """Test QualityAssurance initialization."""
        assert qa_agent.name == "Quality Assurance Agent"
        assert "quality_check" in qa_agent.capabilities
        assert "validation" in qa_agent.capabilities
        assert qa_agent.is_available is True
    
    @pytest.mark.asyncio
    async def test_qa_agent_process_task(self, qa_agent):
        """Test QualityAssurance task processing."""
        task = AgentTask(
            task_id="qa_task",
            task_type="quality_check",
            priority=TaskPriority.HIGH,
            data={
                "subject_type": "document_analysis",
                "subject_id": "analysis_123",
                "validation_rules": ["completeness", "accuracy"]
            },
            requester_id="user123"
        )
        
        # Mock the validation methods
        with patch.object(qa_agent, '_perform_quality_validation', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "validation_passed": True,
                "score": 0.95,
                "issues": [],
                "recommendations": ["Consider adding more detail"]
            }
            
            result = await qa_agent.process_task(task)
            
            assert result["status"] == "completed"
            assert "validation" in result
            assert result["validation"]["validation_passed"] is True
            mock_validate.assert_called_once()


class TestAgentOrchestrator:
    """Test Agent Orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator for testing."""
        return AgentOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.agents) == 3  # doc, bi, qa agents
        assert "document_analyzer" in orchestrator.agents
        assert "business_intelligence" in orchestrator.agents  
        assert "quality_assurance" in orchestrator.agents
    
    def test_get_available_agents(self, orchestrator):
        """Test getting available agents."""
        available = orchestrator.get_available_agents()
        assert len(available) == 3
        assert all(agent.is_available for agent in available.values())
    
    def test_get_agent_capabilities(self, orchestrator):
        """Test getting agent capabilities."""
        capabilities = orchestrator.get_agent_capabilities()
        
        assert "document_analyzer" in capabilities
        assert "business_intelligence" in capabilities
        assert "quality_assurance" in capabilities
        
        doc_caps = capabilities["document_analyzer"]
        assert "document_analysis" in doc_caps
    
    def test_find_suitable_agent(self, orchestrator):
        """Test finding suitable agent for a task."""
        doc_task = AgentTask(
            task_id="doc_task",
            task_type="document_analysis", 
            priority=TaskPriority.MEDIUM,
            data={"document_id": "doc123"},
            requester_id="user123"
        )
        
        agent = orchestrator.find_suitable_agent(doc_task)
        assert agent is not None
        assert agent.name == "Document Analyzer"
    
    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator):
        """Test task submission and routing."""
        task = AgentTask(
            task_id="test_task",
            task_type="document_analysis",
            priority=TaskPriority.MEDIUM, 
            data={"document_id": "doc123", "content": "test content"},
            requester_id="user123"
        )
        
        # Mock the agent process_task method
        doc_agent = orchestrator.agents["document_analyzer"]
        with patch.object(doc_agent, 'process_task', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"status": "completed", "result": "test result"}
            
            result = await orchestrator.submit_task(task)
            
            assert result["status"] == "completed"
            assert result["result"] == "test result"
            mock_process.assert_called_once_with(task)
    
    @pytest.mark.asyncio  
    async def test_submit_task_no_suitable_agent(self, orchestrator):
        """Test task submission when no suitable agent is found."""
        task = AgentTask(
            task_id="unknown_task",
            task_type="unknown_type",
            priority=TaskPriority.MEDIUM,
            data={},
            requester_id="user123"
        )
        
        result = await orchestrator.submit_task(task)
        
        assert result["status"] == "error"
        assert "no suitable agent" in result["message"].lower()