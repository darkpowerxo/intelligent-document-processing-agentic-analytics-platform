"""Example usage of the AI Architect Demo agentic system.

This script demonstrates how to use the multi-agent AI system for
real-world document processing and analysis tasks.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ai_architect_demo.agents.orchestrator import AgentOrchestrator
from ai_architect_demo.agents.document_analyzer import DocumentAnalyzerAgent
from ai_architect_demo.agents.business_intelligence import BusinessIntelligenceAgent
from ai_architect_demo.agents.quality_assurance import QualityAssuranceAgent
from ai_architect_demo.agents.base_agent import AgentTask, TaskPriority


async def simple_document_analysis():
    """Simple example: analyze a business document."""
    print("🚀 Simple Document Analysis Example")
    print("-" * 50)
    
    # Create and setup system
    orchestrator = AgentOrchestrator()
    doc_analyzer = DocumentAnalyzerAgent()
    
    await orchestrator.register_agent(doc_analyzer)
    await orchestrator.start_processing()
    
    # Sample document
    document_text = """
    Monthly Sales Report - January 2024
    
    Sales Performance:
    - Total Revenue: $125,000 (15% increase from December)
    - New Customers: 45 (20% increase)
    - Customer Retention: 92%
    - Top Product: Premium Software License ($45,000 revenue)
    
    Challenges:
    - Higher customer acquisition costs
    - Increased competition in enterprise segment
    - Supply chain delays affecting delivery times
    
    Next Month Goals:
    - Target revenue: $140,000
    - Focus on customer retention programs  
    - Launch new marketing campaign
    """
    
    # Submit and wait for result
    print("📄 Analyzing sales report...")
    task_id = await orchestrator.submit_task(
        task_type="document_analysis",
        data={
            "content": document_text,
            "document_type": "sales_report"
        },
        priority=TaskPriority.HIGH
    )
    
    # Wait a moment for processing
    await asyncio.sleep(2)
    
    # Get task status
    task_status = await orchestrator.get_task_status(task_id)
    
    print(f"✅ Analysis completed!")
    print(f"📊 Task Status: {task_status}")
    
    # Cleanup
    await orchestrator.shutdown()
    await doc_analyzer.cleanup()
    
    return result


async def multi_agent_workflow():
    """Advanced example: multi-agent document processing workflow."""
    print("\n🤖 Multi-Agent Workflow Example")
    print("-" * 50)
    
    # Setup complete system
    orchestrator = AgentOrchestrator()
    
    agents = {
        "analyzer": DocumentAnalyzerAgent(),
        "bi_analyst": BusinessIntelligenceAgent(),
        "qa_specialist": QualityAssuranceAgent()
    }
    
    # Register all agents
    for agent in agents.values():
        await orchestrator.register_agent(agent)
    
    await orchestrator.start_processing()
    
    # Complex business document
    business_document = """
    Q4 2024 Financial Performance Review
    
    Executive Summary:
    We closed Q4 2024 with exceptional results, exceeding our revenue targets
    by 12% and achieving our highest quarterly profit margin at 22.5%.
    
    Financial Highlights:
    - Revenue: $2.8M (34% YoY growth)
    - Gross Profit: $1.9M (38% YoY growth) 
    - Operating Profit: $630K (22.5% margin)
    - Cash Flow: $545K positive
    - R&D Investment: $280K (10% of revenue)
    
    Market Analysis:
    Our core market expanded significantly with three major enterprise clients
    signed in December. Competition remains intense but our differentiated
    AI-powered solutions continue to win deals.
    
    Operational Metrics:
    - Customer Satisfaction: 4.7/5.0 (industry leading)
    - Employee Retention: 94% (up from 89% in Q3)
    - Product Uptime: 99.97% (exceeded SLA)
    - Support Response Time: 2.1 hours average
    
    Strategic Initiatives:
    1. International Expansion: Opened European office
    2. Product Innovation: Launched AI Analytics Suite
    3. Partnership Program: 15 new channel partners
    4. Team Growth: Hired 23 new employees
    
    Risks and Challenges:
    - Economic uncertainty affecting enterprise spending
    - Talent acquisition costs increasing 25%
    - Regulatory changes in data privacy laws
    - Supply chain disruptions in hardware components
    
    Q1 2025 Outlook:
    We project continued growth with revenue target of $3.2M and plan to
    launch our mobile platform while expanding our European operations.
    """
    
    print("📄 Step 1: Document Analysis...")
    # Step 1: Analyze document
    doc_task = AgentTask(
        task_id=f"doc_analysis_{int(datetime.now().timestamp())}",
        task_type="document_analysis", 
        priority=TaskPriority.HIGH,
        data={
            "content": business_document,
            "document_type": "financial_report"
        }
    )
    
    doc_result = await orchestrator.submit_task(doc_task)
    await asyncio.sleep(3)  # Allow processing time
    
    print("📊 Step 2: Business Intelligence Analysis...")
    # Step 2: Business intelligence analysis
    bi_task = AgentTask(
        task_id=f"bi_analysis_{int(datetime.now().timestamp())}",
        task_type="data_analysis",
        priority=TaskPriority.HIGH,
        data={
            "content": business_document,
            "focus_areas": ["financial_performance", "growth_metrics", "risk_analysis"]
        }
    )
    
    bi_result = await orchestrator.submit_task(bi_task)
    await asyncio.sleep(3)
    
    print("🔍 Step 3: Quality Assurance Validation...")
    # Step 3: Quality assurance validation
    qa_task = AgentTask(
        task_id=f"qa_validation_{int(datetime.now().timestamp())}",
        task_type="quality_validation",
        priority=TaskPriority.HIGH,
        data={
            "content": business_document,
            "analysis_results": [doc_result, bi_result],
            "validation_criteria": ["data_accuracy", "completeness", "compliance"]
        }
    )
    
    qa_result = await orchestrator.submit_task(qa_task)
    await asyncio.sleep(3)
    
    # Get final results
    print("✅ Workflow completed! Getting results...")
    
    doc_status = await orchestrator.get_task_status(doc_task.task_id)
    bi_status = await orchestrator.get_task_status(bi_task.task_id) 
    qa_status = await orchestrator.get_task_status(qa_task.task_id)
    
    results = {
        "document_analysis": doc_status,
        "business_intelligence": bi_status,
        "quality_assurance": qa_status,
        "workflow_timestamp": datetime.now().isoformat()
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"workflow_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Results saved to: {results_file}")
    
    # Display summary
    print(f"\n🏆 Workflow Summary:")
    print(f"📄 Document Analysis: {doc_status}")
    print(f"📊 Business Intelligence: {bi_status}")  
    print(f"🔍 Quality Assurance: {qa_status}")
    
    # Cleanup
    await orchestrator.shutdown()
    for agent in agents.values():
        await agent.cleanup()
    
    return results


async def batch_document_processing():
    """Example: process multiple documents in batch."""
    print("\n📦 Batch Document Processing Example")
    print("-" * 50)
    
    # Setup system
    orchestrator = AgentOrchestrator()
    doc_analyzer = DocumentAnalyzerAgent()
    
    await orchestrator.register_agent(doc_analyzer)
    await orchestrator.start_processing()
    
    # Sample documents for batch processing
    documents = {
        "contract_001": "Service Agreement between Company A and Company B for software development services worth $50,000 over 6 months.",
        "invoice_002": "Invoice #INV-2024-001 for Professional Services rendered in January 2024. Amount due: $15,750. Payment terms: Net 30 days.",
        "memo_003": "Internal memo regarding new remote work policy implementation. All employees eligible for hybrid work arrangements starting March 1st.",
        "proposal_004": "Project proposal for AI integration in customer service operations. Estimated cost: $85,000. Timeline: 4 months implementation."
    }
    
    print(f"📄 Processing {len(documents)} documents...")
    
    # Submit all tasks
    tasks = []
    for doc_id, content in documents.items():
        task = AgentTask(
            task_id=f"batch_{doc_id}_{int(datetime.now().timestamp())}",
            task_type="document_analysis",
            priority=TaskPriority.MEDIUM,
            data={
                "content": content,
                "document_type": "business_document",
                "document_id": doc_id
            }
        )
        tasks.append(task)
        await orchestrator.submit_task(task)
    
    # Wait for all processing to complete
    print("⏳ Waiting for batch processing to complete...")
    await asyncio.sleep(5)
    
    # Get all results
    batch_results = {}
    for task in tasks:
        status = await orchestrator.get_task_status(task.task_id)
        doc_id = task.data["document_id"]
        batch_results[doc_id] = status
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = f"batch_results_{timestamp}.json"
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Batch processing completed!")
    print(f"📄 Processed: {len(documents)} documents")
    print(f"📋 Results saved to: {batch_file}")
    
    # Cleanup
    await orchestrator.shutdown()
    await doc_analyzer.cleanup()
    
    return batch_results


async def main():
    """Run all examples."""
    print("🎯 AI Architect Demo - Usage Examples")
    print("=" * 80)
    
    try:
        # Run examples
        await simple_document_analysis()
        await multi_agent_workflow()
        await batch_document_processing()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\n💡 Key Takeaways:")
        print("• Simple single-agent document analysis")
        print("• Complex multi-agent workflows with coordination")
        print("• Batch processing for multiple documents")
        print("• Real-time task status monitoring")
        print("• Comprehensive result reporting")
        print("• Proper resource cleanup")
        
        print("\n🚀 Next Steps:")
        print("• Integrate with your document management system")
        print("• Customize agents for your specific use cases")
        print("• Scale up with additional agent instances")
        print("• Deploy Ollama for enhanced LLM capabilities")
        print("• Add webhook notifications for real-time updates")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())