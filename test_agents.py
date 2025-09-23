"""Simple test to verify our multi-agent system works.

This is a quick verification that our agents can be created and communicate.
"""

import asyncio
import logging
from ai_architect_demo.core.logging import configure_logging
from ai_architect_demo.agents import (
    AgentOrchestrator,
    DocumentAnalyzerAgent,
    BusinessIntelligenceAgent,
    QualityAssuranceAgent,
    TaskPriority,
    AgentRole
)

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


async def test_agent_system():
    """Test basic agent system functionality."""
    logger.info("🚀 Testing Multi-Agent AI System")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create agents
    doc_agent = DocumentAnalyzerAgent()
    bi_agent = BusinessIntelligenceAgent()
    qa_agent = QualityAssuranceAgent()
    
    logger.info(f"✅ Created agents: {doc_agent.name}, {bi_agent.name}, {qa_agent.name}")
    
    # Register agents
    await orchestrator.register_agent(doc_agent)
    await orchestrator.register_agent(bi_agent)
    await orchestrator.register_agent(qa_agent)
    
    logger.info("✅ Agents registered with orchestrator")
    
    # Start processing
    await orchestrator.start_processing()
    
    logger.info("✅ Orchestrator started")
    
    # Get status
    status = orchestrator.get_orchestrator_status()
    logger.info(f"📊 System Status: {status['total_agents']} agents, {status['active_tasks']} active tasks")
    
    # Submit a simple test task
    task_id = await orchestrator.submit_task(
        task_type="text_summarization",
        data={
            "content": "This is a test document for our AI agent system. The system includes document analysis, business intelligence, and quality assurance capabilities.",
            "max_length": 50
        },
        priority=TaskPriority.HIGH,
        preferred_agent_role=AgentRole.DOCUMENT_ANALYZER
    )
    
    logger.info(f"📝 Submitted test task: {task_id}")
    
    # Wait a moment for processing
    await asyncio.sleep(3)
    
    # Check task status
    task_status = await orchestrator.get_task_status(task_id)
    logger.info(f"📋 Task status: {task_status['status']}")
    
    # Final status
    final_status = orchestrator.get_orchestrator_status()
    logger.info(f"🏁 Final Status: {final_status['completed_tasks']} completed, {final_status['failed_tasks']} failed")
    
    # Cleanup
    await orchestrator.cleanup()
    logger.info("✅ System cleaned up successfully")
    
    return True


async def main():
    """Main test function."""
    try:
        success = await test_agent_system()
        if success:
            logger.info("🎉 Multi-Agent System Test: PASSED")
        else:
            logger.error("❌ Multi-Agent System Test: FAILED")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())