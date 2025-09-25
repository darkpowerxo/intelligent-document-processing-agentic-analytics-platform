"""
API routes for agent management and system metrics.
"""

import json
import psutil
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ai_architect_demo.core.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["agents", "system"])

# Response models
class AgentInfo(BaseModel):
    """Agent information model."""
    id: str
    name: str
    type: str
    status: str
    health: str
    current_load: float
    queue_length: int
    tasks_completed: int
    tasks_pending: int
    success_rate: float
    avg_processing_time: float
    last_activity: str
    capabilities: List[str]
    version: str

class AgentsResponse(BaseModel):
    """Response model for agents list."""
    agents: List[AgentInfo]
    total_agents: int
    active_agents: int
    system_health: str

class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    uptime: float
    network_io: Dict[str, float]
    processes: int
    threads: int

class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""
    system: SystemMetrics
    timestamp: datetime
    status: str

# Demo agents data
DEMO_AGENTS = [
    {
        "id": "doc-analyzer-001",
        "name": "Document Analyzer",
        "type": "document_analyzer",
        "status": "active",
        "health": "healthy",
        "current_load": 0.35,
        "queue_length": 12,
        "tasks_completed": 247,
        "tasks_pending": 12,
        "success_rate": 95.2,
        "avg_processing_time": 1.8,
        "last_activity": datetime.now().isoformat(),
        "capabilities": ["pdf_analysis", "text_extraction", "metadata_parsing", "content_classification"],
        "version": "2.1.0"
    },
    {
        "id": "bi-agent-002", 
        "name": "Business Intelligence Agent",
        "type": "business_intelligence",
        "status": "active",
        "health": "healthy",
        "current_load": 0.28,
        "queue_length": 8,
        "tasks_completed": 189,
        "tasks_pending": 8,
        "success_rate": 97.8,
        "avg_processing_time": 2.3,
        "last_activity": datetime.now().isoformat(),
        "capabilities": ["data_analysis", "trend_detection", "report_generation", "dashboard_creation"],
        "version": "1.8.2"
    },
    {
        "id": "qa-agent-003",
        "name": "Quality Assurance Agent", 
        "type": "quality_assurance",
        "status": "active",
        "health": "healthy",
        "current_load": 0.42,
        "queue_length": 15,
        "tasks_completed": 156,
        "tasks_pending": 15,
        "success_rate": 93.7,
        "avg_processing_time": 3.1,
        "last_activity": datetime.now().isoformat(),
        "capabilities": ["data_validation", "accuracy_checking", "compliance_verification", "error_detection"],
        "version": "1.9.1"
    }
]

@router.get("/agents/", response_model=AgentsResponse)
async def list_agents():
    """List all available agents with their status and metrics."""
    try:
        # In a real implementation, this would query actual agent services
        # For demo purposes, we return mock data with some dynamic elements
        
        agents = []
        active_count = 0
        
        for agent_data in DEMO_AGENTS:
            # Add some dynamic variation to make it more realistic
            agent_info = agent_data.copy()
            agent_info["last_activity"] = datetime.now().isoformat()
            
            # Simulate some load variations
            import random
            agent_info["current_load"] = min(0.9, max(0.1, agent_info["current_load"] + random.uniform(-0.1, 0.1)))
            agent_info["queue_length"] = max(0, agent_info["queue_length"] + random.randint(-3, 5))
            
            agents.append(AgentInfo(**agent_info))
            
            if agent_info["status"] == "active":
                active_count += 1
        
        # Determine system health
        avg_load = sum(agent.current_load for agent in agents) / len(agents) if agents else 0
        if avg_load < 0.7:
            system_health = "optimal"
        elif avg_load < 0.85:
            system_health = "warning"
        else:
            system_health = "critical"
        
        return AgentsResponse(
            agents=agents,
            total_agents=len(agents),
            active_agents=active_count,
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agents: {str(e)}"
        )

@router.get("/agents/status")
async def get_agents_status():
    """Get agent system status (simplified endpoint for dashboard)."""
    try:
        agents_response = await list_agents()
        return {
            "agents": [agent.dict() for agent in agents_response.agents],
            "total_agents": agents_response.total_agents,
            "active_agents": agents_response.active_agents,
            "system_health": agents_response.system_health
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )

@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get comprehensive system performance metrics."""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage for root partition
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Get system uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        # Get network I/O
        try:
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": float(net_io.bytes_sent),
                "bytes_recv": float(net_io.bytes_recv),
                "packets_sent": float(net_io.packets_sent),
                "packets_recv": float(net_io.packets_recv)
            }
        except Exception:
            network_io = {
                "bytes_sent": 0.0,
                "bytes_recv": 0.0,
                "packets_sent": 0.0,
                "packets_recv": 0.0
            }
        
        # Get process count
        process_count = len(psutil.pids())
        
        # Get thread count (approximate)
        thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
        
        system_metrics = SystemMetrics(
            cpu_usage=cpu_percent / 100.0,  # Convert to fraction
            memory_usage=memory_percent / 100.0,  # Convert to fraction
            disk_usage=disk_percent / 100.0,  # Convert to fraction
            uptime=uptime_seconds,
            network_io=network_io,
            processes=process_count,
            threads=thread_count
        )
        
        return SystemMetricsResponse(
            system=system_metrics,
            timestamp=datetime.now(),
            status="healthy"
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )

@router.get("/metrics")
async def get_metrics_simple():
    """Simplified metrics endpoint for dashboard compatibility."""
    try:
        system_response = await get_system_metrics()
        return {
            "system": system_response.system.dict(),
            "timestamp": system_response.timestamp.isoformat(),
            "status": system_response.status
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )