"""Business Intelligence Agent for data analysis and insights generation.

This agent specializes in analyzing business data, generating reports,
creating visualizations, and providing strategic insights using AI.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.base_agent import (
    BaseAgent, AgentTask, AgentRole, AgentCapabilities
)

logger = get_logger(__name__)


class BusinessIntelligenceAgent(BaseAgent):
    """Specialized agent for business intelligence and data analysis."""
    
    def __init__(
        self,
        agent_id: str = "bi_agent_001",
        name: str = "Business Intelligence Analyst",
        description: str = "Analyzes business data and generates strategic insights",
        ollama_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1:latest"
    ):
        """Initialize the Business Intelligence Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable agent name
            description: Description of agent's purpose
            ollama_endpoint: Ollama server endpoint
            model_name: LLM model to use
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.BUSINESS_INTELLIGENCE,
            name=name,
            description=description,
            ollama_endpoint=ollama_endpoint,
            model_name=model_name
        )
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Business metrics and KPI templates
        self.kpi_templates = {
            "revenue": self._get_revenue_analysis_template(),
            "customer": self._get_customer_analysis_template(),
            "operational": self._get_operational_analysis_template(),
            "financial": self._get_financial_analysis_template(),
            "marketing": self._get_marketing_analysis_template()
        }
    
    def _define_capabilities(self) -> AgentCapabilities:
        """Define the business intelligence agent's capabilities."""
        return AgentCapabilities(
            supported_tasks=[
                "data_analysis",
                "kpi_calculation",
                "trend_analysis",
                "forecasting",
                "cohort_analysis",
                "customer_segmentation",
                "revenue_analysis",
                "performance_reporting",
                "competitive_analysis",
                "market_analysis",
                "risk_assessment",
                "business_insights",
                "dashboard_generation",
                "anomaly_detection"
            ],
            max_concurrent_tasks=2,
            specializations=[
                "Data Analytics",
                "Business Metrics",
                "Statistical Analysis",
                "Predictive Modeling",
                "Data Visualization",
                "Strategic Planning"
            ],
            required_resources=[
                "local_llm",
                "pandas",
                "matplotlib",
                "seaborn"
            ]
        )
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a business intelligence task.
        
        Args:
            task: Task to process
            
        Returns:
            Analysis results with insights and visualizations
        """
        log_function_call("process_task", agent_id=self.agent_id, task_type=task.task_type)
        
        task_handlers = {
            "data_analysis": self._analyze_business_data,
            "kpi_calculation": self._calculate_kpis,
            "trend_analysis": self._analyze_trends,
            "forecasting": self._generate_forecasts,
            "cohort_analysis": self._perform_cohort_analysis,
            "customer_segmentation": self._segment_customers,
            "revenue_analysis": self._analyze_revenue,
            "performance_reporting": self._generate_performance_report,
            "competitive_analysis": self._analyze_competition,
            "market_analysis": self._analyze_market,
            "risk_assessment": self._assess_risks,
            "business_insights": self._generate_business_insights,
            "dashboard_generation": self._generate_dashboard,
            "anomaly_detection": self._detect_anomalies
        }
        
        handler = task_handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"Unsupported task type: {task.task_type}")
        
        return await handler(task.data)
    
    async def _analyze_business_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive business data analysis.
        
        Args:
            data: Business data to analyze
            
        Returns:
            Analysis results with insights
        """
        logger.info("Starting comprehensive business data analysis")
        
        # Load and prepare data
        df = await self._load_data(data)
        
        if df is None or df.empty:
            return {"error": "Could not load or parse business data"}
        
        # Basic data overview
        analysis_results = {
            "data_overview": {
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": self._get_date_range(df),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        }
        
        # Descriptive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_results["descriptive_stats"] = df[numeric_cols].describe().to_dict()
        
        # Generate insights using LLM
        data_summary = self._create_data_summary(df)
        insights = await self._generate_ai_insights(data_summary, "business_data_analysis")
        analysis_results["ai_insights"] = insights
        
        # Create visualizations
        visualizations = await self._create_basic_visualizations(df)
        analysis_results["visualizations"] = visualizations
        
        logger.info("Business data analysis completed")
        return analysis_results
    
    async def _calculate_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance indicators.
        
        Args:
            data: Data for KPI calculations
            
        Returns:
            Calculated KPIs with analysis
        """
        df = await self._load_data(data)
        kpi_type = data.get("kpi_type", "general")
        
        if df is None:
            return {"error": "Could not load data for KPI calculation"}
        
        # Calculate KPIs based on type
        if kpi_type == "revenue":
            kpis = await self._calculate_revenue_kpis(df)
        elif kpi_type == "customer":
            kpis = await self._calculate_customer_kpis(df)
        elif kpi_type == "operational":
            kpis = await self._calculate_operational_kpis(df)
        else:
            kpis = await self._calculate_general_kpis(df)
        
        # Generate KPI insights
        kpi_summary = self._summarize_kpis(kpis)
        insights = await self._generate_ai_insights(kpi_summary, "kpi_analysis")
        
        return {
            "kpis": kpis,
            "insights": insights,
            "calculation_timestamp": datetime.now().isoformat(),
            "kpi_type": kpi_type
        }
    
    async def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in business data.
        
        Args:
            data: Time series or sequential data
            
        Returns:
            Trend analysis results
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load data for trend analysis"}
        
        # Identify time columns
        time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        metric_cols = df.select_dtypes(include=[np.number]).columns
        
        trends = {}
        
        # Analyze trends for each numeric metric
        for metric in metric_cols:
            if len(time_cols) > 0:
                time_col = time_cols[0]
                trend_data = self._calculate_trend(df, time_col, metric)
                trends[metric] = trend_data
        
        # Generate trend insights
        trend_summary = self._summarize_trends(trends)
        insights = await self._generate_ai_insights(trend_summary, "trend_analysis")
        
        # Create trend visualizations
        visualizations = await self._create_trend_visualizations(df, trends)
        
        return {
            "trends": trends,
            "insights": insights,
            "visualizations": visualizations,
            "analysis_period": self._get_date_range(df)
        }
    
    async def _generate_forecasts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business forecasts.
        
        Args:
            data: Historical data for forecasting
            
        Returns:
            Forecast results with predictions
        """
        df = await self._load_data(data)
        forecast_periods = data.get("forecast_periods", 12)
        
        if df is None:
            return {"error": "Could not load data for forecasting"}
        
        # Simple forecasting using moving averages and trends
        # In production, you'd use more sophisticated models like ARIMA, Prophet, etc.
        
        forecasts = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(df[col].dropna()) > 3:  # Need at least 3 data points
                forecast_data = self._simple_forecast(df[col], forecast_periods)
                forecasts[col] = forecast_data
        
        # Generate forecast insights
        forecast_summary = self._summarize_forecasts(forecasts)
        insights = await self._generate_ai_insights(forecast_summary, "forecasting")
        
        return {
            "forecasts": forecasts,
            "insights": insights,
            "forecast_periods": forecast_periods,
            "confidence_interval": "80%",  # Simplified
            "methodology": "Moving average with trend adjustment"
        }
    
    async def _perform_cohort_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform customer cohort analysis.
        
        Args:
            data: Customer data with dates
            
        Returns:
            Cohort analysis results
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load data for cohort analysis"}
        
        # Simplified cohort analysis
        # In production, you'd need proper customer ID, registration date, and activity data
        
        cohort_data = {
            "message": "Cohort analysis requires customer transaction data with dates",
            "sample_structure": {
                "customer_id": "unique identifier",
                "registration_date": "customer first activity date",
                "transaction_date": "activity dates",
                "revenue": "optional revenue data"
            }
        }
        
        # If we have the right structure, perform actual analysis
        if all(col in df.columns for col in ["customer_id", "date"]):
            cohorts = self._calculate_cohorts(df)
            cohort_data = cohorts
        
        insights = await self._generate_ai_insights(str(cohort_data), "cohort_analysis")
        
        return {
            "cohort_analysis": cohort_data,
            "insights": insights,
            "analysis_type": "customer_retention"
        }
    
    async def _segment_customers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform customer segmentation analysis.
        
        Args:
            data: Customer data for segmentation
            
        Returns:
            Customer segmentation results
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load customer data"}
        
        # Simple segmentation based on available metrics
        segments = {}
        
        # Revenue-based segmentation
        if "revenue" in df.columns:
            revenue_segments = self._segment_by_revenue(df)
            segments["revenue_segments"] = revenue_segments
        
        # Frequency-based segmentation  
        if "frequency" in df.columns or "orders" in df.columns:
            freq_col = "frequency" if "frequency" in df.columns else "orders"
            frequency_segments = self._segment_by_frequency(df, freq_col)
            segments["frequency_segments"] = frequency_segments
        
        # Generate segmentation insights
        segment_summary = self._summarize_segments(segments)
        insights = await self._generate_ai_insights(segment_summary, "customer_segmentation")
        
        return {
            "segments": segments,
            "insights": insights,
            "segmentation_criteria": list(segments.keys()),
            "total_customers": len(df)
        }
    
    async def _analyze_revenue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive revenue analysis.
        
        Args:
            data: Revenue data
            
        Returns:
            Revenue analysis results
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load revenue data"}
        
        # Revenue metrics
        revenue_col = self._find_revenue_column(df)
        
        if not revenue_col:
            return {"error": "Could not identify revenue column in data"}
        
        revenue_analysis = {
            "total_revenue": df[revenue_col].sum(),
            "average_revenue": df[revenue_col].mean(),
            "median_revenue": df[revenue_col].median(),
            "revenue_std": df[revenue_col].std(),
            "max_revenue": df[revenue_col].max(),
            "min_revenue": df[revenue_col].min()
        }
        
        # Time-based revenue analysis if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            monthly_revenue = df.groupby(df[date_col].dt.to_period('M'))[revenue_col].sum()
            revenue_analysis["monthly_revenue"] = monthly_revenue.to_dict()
        
        # Generate revenue insights
        revenue_summary = f"Total revenue: ${revenue_analysis['total_revenue']:,.2f}, Average: ${revenue_analysis['average_revenue']:,.2f}"
        insights = await self._generate_ai_insights(revenue_summary, "revenue_analysis")
        
        return {
            "revenue_metrics": revenue_analysis,
            "insights": insights,
            "currency": data.get("currency", "USD")
        }
    
    async def _generate_performance_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            data: Business performance data
            
        Returns:
            Performance report
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load performance data"}
        
        # Gather key metrics
        performance_metrics = {}
        
        # Revenue performance
        revenue_col = self._find_revenue_column(df)
        if revenue_col:
            performance_metrics["revenue"] = {
                "total": df[revenue_col].sum(),
                "growth_rate": self._calculate_growth_rate(df, revenue_col),
                "trend": self._get_trend_direction(df[revenue_col])
            }
        
        # Customer metrics
        if "customers" in df.columns or "customer_count" in df.columns:
            cust_col = "customers" if "customers" in df.columns else "customer_count"
            performance_metrics["customers"] = {
                "total": df[cust_col].sum(),
                "average_per_period": df[cust_col].mean(),
                "trend": self._get_trend_direction(df[cust_col])
            }
        
        # Generate comprehensive insights
        report_summary = self._create_performance_summary(performance_metrics)
        insights = await self._generate_ai_insights(report_summary, "performance_reporting")
        
        return {
            "performance_metrics": performance_metrics,
            "insights": insights,
            "report_period": self._get_date_range(df),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _analyze_competition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive landscape.
        
        Args:
            data: Competitive data or market data
            
        Returns:
            Competitive analysis results
        """
        # This would typically involve external data sources
        # For demo purposes, we'll analyze provided competitive data
        
        competitor_data = data.get("competitors", [])
        market_data = data.get("market_metrics", {})
        
        competitive_analysis = {
            "market_position": "Analysis requires competitor data",
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": []
        }
        
        # Generate competitive insights using AI
        competitive_context = f"Competitive data: {competitor_data}, Market metrics: {market_data}"
        insights = await self._generate_ai_insights(competitive_context, "competitive_analysis")
        
        return {
            "competitive_analysis": competitive_analysis,
            "insights": insights,
            "analysis_framework": "SWOT Analysis",
            "data_sources": ["internal_data", "market_research"]
        }
    
    async def _analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends and opportunities.
        
        Args:
            data: Market data
            
        Returns:
            Market analysis results
        """
        market_data = data.get("market_data", {})
        industry = data.get("industry", "general")
        
        market_analysis = {
            "market_size": market_data.get("size", "Unknown"),
            "growth_rate": market_data.get("growth_rate", "Unknown"),
            "key_trends": market_data.get("trends", []),
            "market_segments": market_data.get("segments", [])
        }
        
        # Generate market insights
        market_context = f"Industry: {industry}, Market data: {market_data}"
        insights = await self._generate_ai_insights(market_context, "market_analysis")
        
        return {
            "market_analysis": market_analysis,
            "insights": insights,
            "industry": industry,
            "analysis_date": datetime.now().isoformat()
        }
    
    async def _assess_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business risks.
        
        Args:
            data: Business data for risk assessment
            
        Returns:
            Risk assessment results
        """
        df = await self._load_data(data)
        risk_factors = data.get("risk_factors", [])
        
        risk_assessment = {
            "financial_risks": [],
            "operational_risks": [],
            "market_risks": [],
            "regulatory_risks": []
        }
        
        # Analyze data for risk indicators
        if df is not None:
            # Look for volatility indicators
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                volatility = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                if volatility > 0.5:  # High volatility threshold
                    risk_assessment["financial_risks"].append(f"High volatility in {col}")
        
        # Generate risk insights
        risk_context = f"Risk factors: {risk_factors}, Data volatility analysis completed"
        insights = await self._generate_ai_insights(risk_context, "risk_assessment")
        
        return {
            "risk_assessment": risk_assessment,
            "insights": insights,
            "assessment_framework": "Quantitative and Qualitative Analysis",
            "risk_level": "Medium"  # Simplified
        }
    
    async def _generate_business_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate general business insights.
        
        Args:
            data: Business data
            
        Returns:
            Business insights and recommendations
        """
        df = await self._load_data(data)
        business_context = data.get("context", "")
        
        if df is None:
            return {"error": "Could not load business data for insights"}
        
        # Create comprehensive data summary
        data_summary = self._create_comprehensive_summary(df)
        
        # Generate insights using AI
        insights = await self._generate_strategic_insights(data_summary, business_context)
        
        return {
            "insights": insights,
            "data_summary": data_summary,
            "recommendations": insights.get("recommendations", []),
            "next_actions": insights.get("next_actions", [])
        }
    
    async def _generate_dashboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard configuration and visualizations.
        
        Args:
            data: Data for dashboard
            
        Returns:
            Dashboard configuration and charts
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load data for dashboard"}
        
        # Create multiple visualizations
        visualizations = await self._create_comprehensive_visualizations(df)
        
        dashboard_config = {
            "title": data.get("title", "Business Dashboard"),
            "charts": visualizations,
            "refresh_rate": data.get("refresh_rate", "daily"),
            "data_source": "uploaded_data"
        }
        
        return {
            "dashboard": dashboard_config,
            "chart_count": len(visualizations),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in business data.
        
        Args:
            data: Time series or business data
            
        Returns:
            Anomaly detection results
        """
        df = await self._load_data(data)
        
        if df is None:
            return {"error": "Could not load data for anomaly detection"}
        
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Simple statistical anomaly detection
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 10:  # Need sufficient data
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                if len(outliers) > 0:
                    anomalies[col] = {
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(col_data)) * 100,
                        "values": outliers.tolist()
                    }
        
        # Generate anomaly insights
        anomaly_summary = f"Found anomalies in {len(anomalies)} metrics"
        insights = await self._generate_ai_insights(anomaly_summary, "anomaly_detection")
        
        return {
            "anomalies": anomalies,
            "insights": insights,
            "detection_method": "Statistical (IQR-based)",
            "total_anomalies": sum(anom["count"] for anom in anomalies.values())
        }
    
    async def _load_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data into a pandas DataFrame.
        
        Args:
            data: Data source information
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            if "dataframe" in data:
                return pd.DataFrame(data["dataframe"])
            elif "csv_data" in data:
                from io import StringIO
                return pd.read_csv(StringIO(data["csv_data"]))
            elif "file_path" in data:
                if data["file_path"].endswith('.csv'):
                    return pd.read_csv(data["file_path"])
                elif data["file_path"].endswith('.json'):
                    return pd.read_json(data["file_path"])
            elif "records" in data:
                return pd.DataFrame(data["records"])
            else:
                logger.warning("No recognizable data format found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create a text summary of the DataFrame.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Text summary
        """
        summary_parts = [
            f"Dataset contains {len(df)} records with {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}",
            f"Date range: {self._get_date_range(df)}",
            f"Missing values: {df.isnull().sum().sum()} total"
        ]
        
        # Add numeric column summaries
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # First 5 numeric columns
                summary_parts.append(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
        
        return ". ".join(summary_parts)
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Extract date range from DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Date range string
        """
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_cols) == 0:
            # Try to find date columns by name
            potential_date_cols = [col for col in df.columns if 'date' in col.lower()]
            if potential_date_cols:
                try:
                    df[potential_date_cols[0]] = pd.to_datetime(df[potential_date_cols[0]])
                    date_cols = [potential_date_cols[0]]
                except:
                    return "Unknown"
        
        if len(date_cols) > 0:
            date_col = date_cols[0]
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        
        return "No date columns found"
    
    async def _generate_ai_insights(self, data_summary: str, analysis_type: str) -> Dict[str, Any]:
        """Generate AI-powered insights for business data.
        
        Args:
            data_summary: Summary of the data
            analysis_type: Type of analysis being performed
            
        Returns:
            AI-generated insights
        """
        prompt = f"""
As a business intelligence expert, analyze the following data summary and provide strategic insights:

Data Summary: {data_summary}
Analysis Type: {analysis_type}

Please provide:
1. Key findings from the data
2. Business implications
3. Strategic recommendations
4. Potential risks or opportunities
5. Next actions to take

Respond in JSON format:
{{
    "key_findings": ["finding1", "finding2", ...],
    "business_implications": ["implication1", "implication2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "risks_opportunities": ["risk1", "opportunity1", ...],
    "next_actions": ["action1", "action2", ...]
}}
"""
        
        system_prompt = "You are a senior business intelligence analyst with expertise in data interpretation and strategic planning."
        
        try:
            return await self.query_llm_structured(
                prompt,
                {
                    "key_findings": [],
                    "business_implications": [],
                    "recommendations": [],
                    "risks_opportunities": [],
                    "next_actions": []
                },
                system_prompt
            )
        except Exception as e:
            logger.error(f"Failed to generate AI insights: {e}")
            return {"error": "Could not generate insights"}
    
    async def _create_basic_visualizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create basic visualizations for the data.
        
        Args:
            df: DataFrame to visualize
            
        Returns:
            List of visualization configurations
        """
        visualizations = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Create histogram for first numeric column
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                plt.figure(figsize=(10, 6))
                plt.hist(df[col].dropna(), bins=20, alpha=0.7)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
                # Convert plot to base64 string
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "image": image_base64,
                    "description": f"Frequency distribution of {col} values"
                })
            
            # Create correlation matrix if multiple numeric columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                visualizations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "image": image_base64,
                    "description": "Correlation between numeric variables"
                })
                
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
        
        return visualizations
    
    # Helper methods for various calculations
    def _find_revenue_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the revenue column in the DataFrame."""
        revenue_keywords = ['revenue', 'sales', 'income', 'earnings', 'amount', 'value', 'price']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in revenue_keywords):
                if df[col].dtype in [np.number]:
                    return col
        
        return None
    
    def _calculate_growth_rate(self, df: pd.DataFrame, column: str) -> float:
        """Calculate growth rate for a column."""
        if len(df) < 2:
            return 0.0
        
        first_value = df[column].iloc[0]
        last_value = df[column].iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _get_trend_direction(self, series: pd.Series) -> str:
        """Determine trend direction of a series."""
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    # Template methods for different analysis types
    def _get_revenue_analysis_template(self) -> str:
        return "Revenue analysis focusing on growth, trends, and profitability metrics"
    
    def _get_customer_analysis_template(self) -> str:
        return "Customer analysis including acquisition, retention, and lifetime value"
    
    def _get_operational_analysis_template(self) -> str:
        return "Operational analysis covering efficiency, productivity, and resource utilization"
    
    def _get_financial_analysis_template(self) -> str:
        return "Financial analysis including ratios, cash flow, and performance indicators"
    
    def _get_marketing_analysis_template(self) -> str:
        return "Marketing analysis covering campaigns, channels, and ROI metrics"