"""Demonstration script for the multi-agent AI system.

This script demonstrates how to use the agent orchestration system
to process documents, generate business intelligence, and ensure quality.
"""

import asyncio
import json
import logging
from typing import Dict, Any
from datetime import datetime

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


class MultiAgentDemo:
    """Demo class for multi-agent system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.orchestrator = AgentOrchestrator()
        self.agents = {}
        
    async def initialize_agents(self) -> None:
        """Initialize and register all agents."""
        logger.info("Initializing agents...")
        
        # Create specialized agents
        doc_agent = DocumentAnalyzerAgent()
        bi_agent = BusinessIntelligenceAgent()
        qa_agent = QualityAssuranceAgent()
        
        # Register agents with orchestrator
        await self.orchestrator.register_agent(doc_agent)
        await self.orchestrator.register_agent(bi_agent)
        await self.orchestrator.register_agent(qa_agent)
        
        # Store references for direct access if needed
        self.agents = {
            "document_analyzer": doc_agent,
            "business_intelligence": bi_agent,
            "quality_assurance": qa_agent
        }
        
        # Start the orchestrator's task processing
        await self.orchestrator.start_processing()
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def demo_document_analysis(self) -> Dict[str, Any]:
        """Demonstrate document analysis capabilities."""
        logger.info("=== Document Analysis Demo ===")
        
        # Sample business document
        sample_document = """
        QUARTERLY BUSINESS REPORT - Q3 2024
        
        Executive Summary:
        This quarter has shown exceptional growth across all key metrics. Revenue increased by 23% 
        compared to Q2 2024, reaching $2.3 million. Customer acquisition improved by 15% with 
        successful campaigns in digital marketing channels.
        
        Key Achievements:
        • Launched new product line with $400K initial sales
        • Expanded into European markets with 3 new partnerships
        • Improved customer satisfaction scores to 4.7/5.0
        • Reduced operational costs by 8% through automation
        
        Challenges:
        • Supply chain delays affected 12% of orders
        • Competition increased pricing pressure in core markets
        • Staff turnover in sales team reached 18%
        
        Action Items:
        1. Negotiate better terms with suppliers by end of Q4
        2. Develop competitive pricing strategy for 2025
        3. Implement retention program for sales staff
        4. Expand digital marketing budget by 25%
        
        Financial Highlights:
        • Revenue: $2,300,000 (↑23% QoQ)
        • Gross Profit: $1,150,000 (↑21% QoQ)  
        • Operating Expenses: $890,000 (↓8% QoQ)
        • Net Income: $260,000 (↑45% QoQ)
        
        Looking ahead to Q4, we anticipate continued growth with the holiday season
        and new product launches expected to drive revenue to $2.8 million.
        """
        
        # Submit document analysis task
        task_data = {
            "content": sample_document,
            "analysis_types": ["summary", "key_points", "entities", "action_items"]
        }
        
        task_id = await self.orchestrator.submit_task(
            task_type="document_analysis",
            data=task_data,
            priority=TaskPriority.HIGH,
            preferred_agent_role=AgentRole.DOCUMENT_ANALYZER
        )
        
        logger.info(f"Submitted document analysis task: {task_id}")
        
        # Wait for completion and get results
        result = await self._wait_for_task_completion(task_id)
        
        if result:
            logger.info("Document Analysis Results:")
            logger.info(f"- Summary length: {len(result.get('summary', {}).get('summary', ''))}")
            logger.info(f"- Key points found: {len(result.get('key_points', []))}")
            logger.info(f"- Entities extracted: {result.get('entities', {}).get('entity_count', 0)}")
            logger.info(f"- Action items: {result.get('action_items', {}).get('total_actions', 0)}")
        
        return result
    
    async def demo_business_intelligence(self) -> Dict[str, Any]:
        """Demonstrate business intelligence capabilities."""
        logger.info("=== Business Intelligence Demo ===")
        
        # Sample business data
        sample_data = {
            "records": [
                {"date": "2024-01-01", "revenue": 180000, "customers": 1200, "orders": 850},
                {"date": "2024-02-01", "revenue": 195000, "customers": 1350, "orders": 920},
                {"date": "2024-03-01", "revenue": 210000, "customers": 1420, "orders": 980},
                {"date": "2024-04-01", "revenue": 198000, "customers": 1380, "orders": 910},
                {"date": "2024-05-01", "revenue": 215000, "customers": 1480, "orders": 1020},
                {"date": "2024-06-01", "revenue": 228000, "customers": 1560, "orders": 1080},
                {"date": "2024-07-01", "revenue": 235000, "customers": 1620, "orders": 1150},
                {"date": "2024-08-01", "revenue": 248000, "customers": 1680, "orders": 1200},
                {"date": "2024-09-01", "revenue": 265000, "customers": 1750, "orders": 1280}
            ]
        }
        
        # Submit business intelligence task
        task_data = {
            "dataframe": sample_data["records"],
            "analysis_types": ["revenue", "customer", "trend"]
        }
        
        task_id = await self.orchestrator.submit_task(
            task_type="data_analysis",
            data=task_data,
            priority=TaskPriority.MEDIUM,
            preferred_agent_role=AgentRole.BUSINESS_INTELLIGENCE
        )
        
        logger.info(f"Submitted business intelligence task: {task_id}")
        
        # Wait for completion and get results
        result = await self._wait_for_task_completion(task_id)
        
        if result:
            overview = result.get("data_overview", {})
            logger.info("Business Intelligence Results:")
            logger.info(f"- Records analyzed: {overview.get('total_records', 0)}")
            logger.info(f"- Date range: {overview.get('date_range', 'Unknown')}")
            logger.info(f"- Visualizations created: {len(result.get('visualizations', []))}")
            
            # Show some key insights
            insights = result.get("ai_insights", {})
            if insights.get("key_findings"):
                logger.info("Key Findings:")
                for finding in insights["key_findings"][:3]:
                    logger.info(f"  • {finding}")
        
        return result
    
    async def demo_quality_assurance(self) -> Dict[str, Any]:
        """Demonstrate quality assurance capabilities."""
        logger.info("=== Quality Assurance Demo ===")
        
        # Sample content for quality validation
        sample_output = {
            "content": """
            Based on the analysis of Q3 2024 business data, here are the key findings:
            
            Revenue Performance: The company achieved $2.3 million in revenue, representing 
            a 23% increase from the previous quarter. This growth is primarily attributed to 
            successful product launches and market expansion.
            
            Customer Metrics: Customer acquisition improved by 15%, with digital marketing 
            campaigns showing particularly strong results. Customer satisfaction scores 
            reached 4.7/5.0, indicating high service quality.
            
            Operational Efficiency: The organization successfully reduced operational costs 
            by 8% through automation initiatives, while maintaining service quality standards.
            
            Recommendations:
            1. Continue investing in digital marketing channels
            2. Expand automation to additional business processes  
            3. Develop strategies to address supply chain challenges
            4. Implement staff retention programs to reduce turnover
            """,
            "metadata": {
                "analysis_type": "quarterly_report",
                "data_sources": ["financial_reports", "customer_surveys", "operational_metrics"],
                "confidence_level": 0.87
            }
        }
        
        # Submit quality validation task
        task_data = {
            "content": sample_output["content"],
            "validation_type": "content_quality",
            "standards": ["content_quality", "output_quality"],
            "criteria": ["accuracy", "clarity", "completeness", "consistency"]
        }
        
        task_id = await self.orchestrator.submit_task(
            task_type="quality_validation",
            data=task_data,
            priority=TaskPriority.HIGH,
            preferred_agent_role=AgentRole.QUALITY_ASSURANCE
        )
        
        logger.info(f"Submitted quality assurance task: {task_id}")
        
        # Wait for completion and get results
        result = await self._wait_for_task_completion(task_id)
        
        if result:
            logger.info("Quality Assurance Results:")
            logger.info(f"- Overall Status: {result.get('overall_status', 'Unknown')}")
            logger.info(f"- Quality Score: {result.get('overall_score', 0):.2f}")
            logger.info(f"- Quality Level: {result.get('quality_level', 'Unknown')}")
            logger.info(f"- Issues Found: {len(result.get('issues', []))}")
            
            # Show quality insights
            if result.get("key_findings"):
                logger.info("Quality Findings:")
                for finding in result["key_findings"][:3]:
                    logger.info(f"  • {finding}")
        
        return result
    
    async def demo_agent_coordination(self) -> Dict[str, Any]:
        """Demonstrate multi-agent coordination."""
        logger.info("=== Agent Coordination Demo ===")
        
        # Submit multiple related tasks simultaneously
        document_task_id = await self.orchestrator.submit_task(
            task_type="text_summarization",
            data={
                "content": "This is a comprehensive business analysis document that requires summarization for executive review.",
                "max_length": 100,
                "style": "executive"
            },
            priority=TaskPriority.MEDIUM,
            preferred_agent_role=AgentRole.DOCUMENT_ANALYZER
        )
        
        analysis_task_id = await self.orchestrator.submit_task(
            task_type="kpi_calculation", 
            data={
                "records": [
                    {"revenue": 100000, "customers": 500},
                    {"revenue": 120000, "customers": 580},
                    {"revenue": 135000, "customers": 620}
                ],
                "kpi_type": "revenue"
            },
            priority=TaskPriority.MEDIUM,
            preferred_agent_role=AgentRole.BUSINESS_INTELLIGENCE
        )
        
        validation_task_id = await self.orchestrator.submit_task(
            task_type="output_validation",
            data={
                "actual_output": "Revenue increased by 17.5% over the analysis period",
                "validation_rules": ["format_check", "accuracy_check"],
                "context": "financial_analysis"
            },
            priority=TaskPriority.LOW,
            preferred_agent_role=AgentRole.QUALITY_ASSURANCE
        )
        
        logger.info(f"Submitted coordinated tasks: {document_task_id}, {analysis_task_id}, {validation_task_id}")
        
        # Wait for all tasks to complete
        results = {}
        for task_name, task_id in [("document", document_task_id), ("analysis", analysis_task_id), ("validation", validation_task_id)]:
            result = await self._wait_for_task_completion(task_id)
            results[task_name] = result
            logger.info(f"Task {task_name} completed: {bool(result)}")
        
        return results
    
    async def show_orchestrator_status(self) -> None:
        """Display orchestrator and agent status."""
        logger.info("=== System Status ===")
        
        status = self.orchestrator.get_orchestrator_status()
        
        logger.info(f"Orchestrator Uptime: {status['orchestrator_uptime']:.1f} seconds")
        logger.info(f"Total Agents: {status['total_agents']}")
        logger.info(f"Active Tasks: {status['active_tasks']}")
        logger.info(f"Completed Tasks: {status['completed_tasks']}")
        logger.info(f"Failed Tasks: {status['failed_tasks']}")
        
        if status['total_tasks_processed'] > 0:
            logger.info(f"Average Processing Time: {status['average_processing_time']:.2f}s")
        
        # Show individual agent statuses
        logger.info("Agent Status:")
        for agent_id, agent_status in status['agent_statuses'].items():
            agent_name = agent_status['name']
            agent_role = agent_status['role']
            current_tasks = agent_status['current_tasks']
            completed_tasks = agent_status['completed_tasks']
            success_rate = agent_status['success_rate']
            
            logger.info(f"  {agent_name} ({agent_role}): {current_tasks} active, {completed_tasks} completed, {success_rate:.1%} success rate")
    
    async def _wait_for_task_completion(self, task_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for a task to complete and return its result.
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            Task result or None if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            status = await self.orchestrator.get_task_status(task_id)
            
            if status["status"] == "completed":
                return status.get("result")
            elif status["status"] == "failed":
                logger.error(f"Task {task_id} failed: {status.get('error', 'Unknown error')}")
                return None
            
            await asyncio.sleep(1.0)  # Check every second
        
        logger.warning(f"Task {task_id} timed out after {timeout} seconds")
        return None
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        await self.orchestrator.cleanup()


async def main():
    """Main demo function."""
    logger.info("Starting Multi-Agent AI System Demo")
    logger.info("=" * 50)
    
    demo = MultiAgentDemo()
    
    try:
        # Initialize the system
        await demo.initialize_agents()
        await demo.show_orchestrator_status()
        
        # Run demonstrations
        logger.info("\nRunning demonstrations...")
        
        # Document Analysis Demo
        doc_result = await demo.demo_document_analysis()
        await asyncio.sleep(2)  # Brief pause between demos
        
        # Business Intelligence Demo  
        bi_result = await demo.demo_business_intelligence()
        await asyncio.sleep(2)
        
        # Quality Assurance Demo
        qa_result = await demo.demo_quality_assurance()
        await asyncio.sleep(2)
        
        # Agent Coordination Demo
        coord_result = await demo.demo_agent_coordination()
        
        # Final status
        await demo.show_orchestrator_status()
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        logger.info(f"Document Analysis: {'✓' if doc_result else '✗'}")
        logger.info(f"Business Intelligence: {'✓' if bi_result else '✗'}")
        logger.info(f"Quality Assurance: {'✓' if qa_result else '✗'}")
        logger.info(f"Agent Coordination: {'✓' if coord_result else '✗'}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())