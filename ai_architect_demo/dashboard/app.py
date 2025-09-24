#!/usr/bin/env python3
"""
Streamlit Web Dashboard for AI Architect Demo
Enterprise-Grade Interface for Document Processing & Agentic Analytics
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Any

# Page configuration
st.set_page_config(
    page_title="AI Architecture Demo Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class DashboardAPI:
    """API client for dashboard data"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/metrics", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent system status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/agents/status", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming system statistics"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/streaming/stats", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}

def main():
    """Main dashboard application"""
    
    # Initialize API client
    api = DashboardAPI()
    
    # Header
    st.title("🤖 AI Architecture Demo Dashboard")
    st.markdown("**Enterprise Document Processing & Agentic Analytics Platform**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        # System status
        st.header("🔍 System Status")
        health = api.health_check()
        
        if health.get("status") == "healthy":
            st.success("🟢 System Healthy")
        else:
            st.error("🔴 System Error")
            if "message" in health:
                st.error(f"Error: {health['message']}")
        
        # Navigation
        st.header("📊 Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "Agent Analytics", "Streaming Monitor", "System Metrics", "Document Processing"]
        )
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(api)
    elif page == "Agent Analytics":
        show_agent_analytics(api)
    elif page == "Streaming Monitor":
        show_streaming_monitor(api)
    elif page == "System Metrics":
        show_system_metrics(api)
    elif page == "Document Processing":
        show_document_processing(api)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def show_overview(api: DashboardAPI):
    """Show system overview dashboard"""
    st.header("📈 System Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Active Agents",
            value="3",
            delta="All Online"
        )
    
    with col2:
        st.metric(
            label="📄 Documents Processed",
            value="1,247",
            delta="+23 today"
        )
    
    with col3:
        st.metric(
            label="⚡ Avg Response Time",
            value="1.2s",
            delta="-0.3s"
        )
    
    with col4:
        st.metric(
            label="🚀 System Uptime",
            value="99.8%",
            delta="Excellent"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Processing Volume")
        # Sample data for demo
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        volumes = [120, 150, 180, 220, 190, 250, 280, 300]  # Added one more value to match dates length
        
        fig = px.line(
            x=dates, 
            y=volumes,
            title="Daily Document Processing Volume",
            labels={"x": "Date", "y": "Documents"}
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🎯 Agent Performance")
        agents = ["Document Analyzer", "Business Intelligence", "Quality Assurance"]
        performance = [95.2, 97.8, 93.4]
        
        fig = px.bar(
            x=agents,
            y=performance,
            title="Agent Success Rates (%)",
            color=performance,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, width='stretch')
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    activity_data = {
        "Timestamp": [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=12),
            datetime.now() - timedelta(minutes=18),
            datetime.now() - timedelta(minutes=25)
        ],
        "Event": [
            "Document analysis completed",
            "Business insight generated",
            "Quality check passed",
            "New document uploaded"
        ],
        "Status": ["✅ Success", "✅ Success", "✅ Success", "📤 Processing"],
        "Agent": ["Document Analyzer", "BI Agent", "QA Agent", "System"]
    }
    
    df = pd.DataFrame(activity_data)
    st.dataframe(df, width='stretch')

def show_agent_analytics(api: DashboardAPI):
    """Show agent analytics dashboard"""
    st.header("🤖 Agent Analytics")
    
    # Agent status
    agent_status = api.get_agent_status()
    
    if agent_status:
        col1, col2, col3 = st.columns(3)
        
        agents = [
            {"name": "Document Analyzer", "status": "active", "tasks": 45},
            {"name": "Business Intelligence", "status": "active", "tasks": 32},
            {"name": "Quality Assurance", "status": "active", "tasks": 28}
        ]
        
        for i, agent in enumerate(agents):
            col = [col1, col2, col3][i]
            with col:
                st.metric(
                    label=f"🎯 {agent['name']}",
                    value=f"{agent['tasks']} tasks",
                    delta="Active" if agent['status'] == 'active' else "Inactive"
                )
    
    # Performance metrics
    st.subheader("📈 Performance Trends")
    
    # Sample performance data
    hours = list(range(24))
    doc_analyzer = [20 + i * 2 + (i % 3) * 5 for i in range(24)]
    bi_agent = [15 + i * 1.5 + (i % 4) * 3 for i in range(24)]
    qa_agent = [10 + i * 1.8 + (i % 2) * 4 for i in range(24)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=doc_analyzer, name="Document Analyzer", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=hours, y=bi_agent, name="Business Intelligence", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=hours, y=qa_agent, name="Quality Assurance", mode="lines+markers"))
    
    fig.update_layout(
        title="Agent Task Processing (24h)",
        xaxis_title="Hour of Day",
        yaxis_title="Tasks Processed"
    )
    
    st.plotly_chart(fig, width='stretch')

def show_streaming_monitor(api: DashboardAPI):
    """Show streaming system monitor"""
    st.header("🌊 Streaming Monitor")
    
    # Streaming stats
    streaming_stats = api.get_streaming_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📬 Messages/sec", "847", "+12%")
    
    with col2:
        st.metric("🔄 Topic Count", "8", "Active")
    
    with col3:
        st.metric("⚡ Latency", "23ms", "-5ms")
    
    # Topic overview
    st.subheader("📋 Kafka Topics")
    
    topics_data = {
        "Topic": ["document-events", "analysis-results", "bi-insights", "qa-reports", "system-metrics"],
        "Messages": [1247, 892, 445, 334, 2156],
        "Partitions": [3, 2, 2, 1, 4],
        "Consumers": [2, 1, 1, 1, 3],
        "Status": ["🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟢 Healthy"]
    }
    
    st.dataframe(pd.DataFrame(topics_data), width='stretch')

def show_system_metrics(api: DashboardAPI):
    """Show system metrics dashboard"""
    st.header("🖥️ System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💾 Memory Usage", "68%", "+2%")
    
    with col2:
        st.metric("⚡ CPU Usage", "45%", "-3%")
    
    with col3:
        st.metric("💿 Disk Usage", "34%", "+1%")
    
    with col4:
        st.metric("🌐 Network I/O", "2.4 MB/s", "+15%")
    
    # System health chart
    st.subheader("📊 System Health Trends")
    
    # Sample data
    times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    cpu_data = [30 + i * 0.5 + (i % 6) * 8 for i in range(len(times))]
    memory_data = [60 + i * 0.3 + (i % 4) * 5 for i in range(len(times))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=cpu_data, name="CPU %", mode="lines"))
    fig.add_trace(go.Scatter(x=times, y=memory_data, name="Memory %", mode="lines"))
    
    fig.update_layout(
        title="System Resource Usage (24h)",
        xaxis_title="Time",
        yaxis_title="Usage %"
    )
    
    st.plotly_chart(fig, width='stretch')

def show_document_processing(api: DashboardAPI):
    """Show document processing interface"""
    st.header("📄 Document Processing")
    
    # Upload section
    st.subheader("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose files to process",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'csv', 'xlsx']
    )
    
    if uploaded_file:
        st.success(f"Selected {len(uploaded_file)} file(s) for processing")
        
        if st.button("🚀 Start Processing", type="primary"):
            st.success("Processing started! Results will appear in the activity feed.")
    
    # Processing queue
    st.subheader("⏳ Processing Queue")
    
    queue_data = {
        "File": ["report_q3.pdf", "analysis_data.xlsx", "summary.docx"],
        "Size": ["2.4 MB", "856 KB", "124 KB"],
        "Status": ["🔄 Processing", "⏳ Queued", "⏳ Queued"],
        "Progress": [75, 0, 0],
        "ETA": ["2 min", "5 min", "7 min"]
    }
    
    st.dataframe(pd.DataFrame(queue_data), width='stretch')
    
    # Results
    st.subheader("✅ Recent Results")
    
    results_data = {
        "Document": ["monthly_report.pdf", "customer_data.csv", "presentation.pptx"],
        "Processed": ["5 min ago", "15 min ago", "1 hour ago"],
        "Insights": [12, 8, 5],
        "Quality Score": [95.2, 87.6, 91.4],
        "Action": ["📊 View", "📊 View", "📊 View"]
    }
    
    st.dataframe(pd.DataFrame(results_data), width='stretch')

if __name__ == "__main__":
    main()