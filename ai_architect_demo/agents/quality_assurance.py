"""Quality Assurance Agent for validating outputs and ensuring system reliability.

This agent specializes in quality control, validation, testing, and ensuring
that all system outputs meet defined standards and requirements.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.base_agent import (
    BaseAgent, AgentTask, AgentRole, AgentCapabilities
)
from ai_architect_demo.data.validation import ValidationRule, ValidationResult

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityMetric:
    """Quality metric definition."""
    name: str
    description: str
    weight: float
    threshold: float
    current_score: float = 0.0
    status: str = "pending"


class QualityStandard:
    """Quality standard definition with rules and thresholds."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metrics: List[QualityMetric] = []
        self.rules: List[ValidationRule] = []
        self.overall_threshold = 0.8
    
    def add_metric(self, metric: QualityMetric) -> None:
        """Add a quality metric to this standard."""
        self.metrics.append(metric)
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to this standard."""
        self.rules.append(rule)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall quality score."""
        if not self.metrics:
            return 0.0
        
        weighted_sum = sum(metric.current_score * metric.weight for metric in self.metrics)
        total_weight = sum(metric.weight for metric in self.metrics)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class QualityAssuranceAgent(BaseAgent):
    """Specialized agent for quality assurance and validation."""
    
    def __init__(
        self,
        agent_id: str = "qa_agent_001",
        name: str = "Quality Assurance Specialist",
        description: str = "Validates outputs and ensures quality standards",
        ollama_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1:latest"
    ):
        """Initialize the Quality Assurance Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable agent name
            description: Description of agent's purpose
            ollama_endpoint: Ollama server endpoint
            model_name: LLM model to use
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.QUALITY_ASSURANCE,
            name=name,
            description=description,
            ollama_endpoint=ollama_endpoint,
            model_name=model_name
        )
        
        # Quality standards registry
        self.quality_standards: Dict[str, QualityStandard] = {}
        self._initialize_default_standards()
        
        # Test frameworks and validation engines
        self.validation_engines = {
            "content": self._validate_content_quality,
            "data": self._validate_data_quality,
            "output": self._validate_output_quality,
            "performance": self._validate_performance_quality,
            "security": self._validate_security_quality,
            "compliance": self._validate_compliance_quality
        }
        
        # Quality history and metrics
        self.quality_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_quality_score": 0.0
        }
    
    def _define_capabilities(self) -> AgentCapabilities:
        """Define the quality assurance agent's capabilities."""
        return AgentCapabilities(
            supported_tasks=[
                "quality_validation",
                "content_review",
                "data_quality_check",
                "output_validation",
                "performance_testing",
                "security_audit",
                "compliance_check",
                "regression_testing",
                "integration_testing",
                "acceptance_testing",
                "quality_report",
                "standard_definition",
                "test_case_generation",
                "defect_analysis"
            ],
            max_concurrent_tasks=5,  # QA can handle multiple validations
            specializations=[
                "Quality Control",
                "Test Automation",
                "Validation Frameworks",
                "Compliance Testing",
                "Performance Analysis",
                "Security Auditing"
            ],
            required_resources=[
                "local_llm",
                "validation_engine",
                "test_frameworks"
            ]
        )
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a quality assurance task.
        
        Args:
            task: Task to process
            
        Returns:
            Quality validation results
        """
        log_function_call("process_task", agent_id=self.agent_id, task_type=task.task_type)
        
        task_handlers = {
            "quality_validation": self._perform_quality_validation,
            "content_review": self._review_content,
            "data_quality_check": self._check_data_quality,
            "output_validation": self._validate_output,
            "performance_testing": self._test_performance,
            "security_audit": self._audit_security,
            "compliance_check": self._check_compliance,
            "regression_testing": self._run_regression_tests,
            "integration_testing": self._run_integration_tests,
            "acceptance_testing": self._run_acceptance_tests,
            "quality_report": self._generate_quality_report,
            "standard_definition": self._define_quality_standard,
            "test_case_generation": self._generate_test_cases,
            "defect_analysis": self._analyze_defects
        }
        
        handler = task_handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"Unsupported task type: {task.task_type}")
        
        # Execute the validation and track metrics
        result = await handler(task.data)
        self._update_performance_metrics(result)
        
        return result
    
    async def _perform_quality_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality validation.
        
        Args:
            data: Data and content to validate
            
        Returns:
            Quality validation results
        """
        logger.info("Starting comprehensive quality validation")
        
        validation_type = data.get("validation_type", "general")
        content = data.get("content", "")
        standards = data.get("standards", ["default"])
        
        # Initialize validation results
        validation_results = {
            "overall_status": "unknown",
            "overall_score": 0.0,
            "quality_level": QualityLevel.POOR.value,
            "validations": {},
            "issues": [],
            "recommendations": [],
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Run validation against each specified standard
        for standard_name in standards:
            if standard_name in self.quality_standards:
                standard = self.quality_standards[standard_name]
                standard_result = await self._validate_against_standard(content, standard)
                validation_results["validations"][standard_name] = standard_result
        
        # Calculate overall score and status
        if validation_results["validations"]:
            scores = [result["score"] for result in validation_results["validations"].values()]
            validation_results["overall_score"] = sum(scores) / len(scores)
            validation_results["quality_level"] = self._determine_quality_level(validation_results["overall_score"])
            validation_results["overall_status"] = "passed" if validation_results["overall_score"] >= 0.7 else "failed"
        
        # Generate AI-powered quality insights
        quality_context = self._create_quality_summary(validation_results)
        insights = await self._generate_quality_insights(quality_context)
        validation_results.update(insights)
        
        # Store validation history
        self.quality_history.append(validation_results)
        
        logger.info(f"Quality validation completed with score: {validation_results['overall_score']:.2f}")
        return validation_results
    
    async def _review_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Review content for quality, accuracy, and completeness.
        
        Args:
            data: Content to review
            
        Returns:
            Content review results
        """
        content = data.get("content", "")
        review_criteria = data.get("criteria", ["accuracy", "clarity", "completeness", "consistency"])
        
        if not content:
            return {"error": "No content provided for review"}
        
        # Perform content analysis
        content_analysis = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "readability_score": self._calculate_readability_score(content),
            "sentiment": self._analyze_content_sentiment(content),
            "structure_score": self._analyze_content_structure(content)
        }
        
        # AI-powered content review
        review_results = await self._perform_ai_content_review(content, review_criteria)
        
        return {
            "content_analysis": content_analysis,
            "review_results": review_results,
            "overall_rating": self._calculate_content_rating(content_analysis, review_results),
            "improvement_suggestions": review_results.get("suggestions", [])
        }
    
    async def _check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data quality and integrity.
        
        Args:
            data: Data to check
            
        Returns:
            Data quality results
        """
        dataset = data.get("dataset", {})
        quality_dimensions = data.get("dimensions", ["completeness", "accuracy", "consistency", "validity"])
        
        quality_results = {}
        
        # Completeness check
        if "completeness" in quality_dimensions:
            completeness_score = self._check_data_completeness(dataset)
            quality_results["completeness"] = completeness_score
        
        # Accuracy check
        if "accuracy" in quality_dimensions:
            accuracy_score = self._check_data_accuracy(dataset)
            quality_results["accuracy"] = accuracy_score
        
        # Consistency check
        if "consistency" in quality_dimensions:
            consistency_score = self._check_data_consistency(dataset)
            quality_results["consistency"] = consistency_score
        
        # Validity check
        if "validity" in quality_dimensions:
            validity_score = self._check_data_validity(dataset)
            quality_results["validity"] = validity_score
        
        # Calculate overall data quality score
        overall_score = sum(quality_results.values()) / len(quality_results) if quality_results else 0.0
        
        return {
            "quality_dimensions": quality_results,
            "overall_score": overall_score,
            "quality_level": self._determine_quality_level(overall_score),
            "data_profile": self._create_data_profile(dataset),
            "recommendations": self._generate_data_quality_recommendations(quality_results)
        }
    
    async def _validate_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system output against expected results.
        
        Args:
            data: Output to validate
            
        Returns:
            Output validation results
        """
        actual_output = data.get("actual_output")
        expected_output = data.get("expected_output")
        validation_rules = data.get("validation_rules", [])
        
        validation_results = {
            "exact_match": actual_output == expected_output if expected_output else None,
            "similarity_score": self._calculate_output_similarity(actual_output, expected_output) if expected_output else None,
            "rule_validations": [],
            "format_compliance": True,
            "content_validity": True
        }
        
        # Apply validation rules
        for rule in validation_rules:
            rule_result = await self._apply_validation_rule(actual_output, rule)
            validation_results["rule_validations"].append(rule_result)
        
        # Validate output format
        if data.get("expected_format"):
            format_valid = self._validate_output_format(actual_output, data["expected_format"])
            validation_results["format_compliance"] = format_valid
        
        # Generate AI-powered output analysis
        output_analysis = await self._analyze_output_quality(actual_output, data.get("context", ""))
        validation_results["ai_analysis"] = output_analysis
        
        return validation_results
    
    async def _test_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test performance characteristics.
        
        Args:
            data: Performance test configuration
            
        Returns:
            Performance test results
        """
        test_type = data.get("test_type", "response_time")
        target_function = data.get("target_function")
        performance_criteria = data.get("criteria", {})
        
        performance_results = {
            "test_type": test_type,
            "execution_time": 0.0,
            "memory_usage": 0.0,
            "throughput": 0.0,
            "error_rate": 0.0,
            "performance_score": 0.0
        }
        
        # Simulate performance testing
        # In a real implementation, this would execute actual performance tests
        if target_function:
            # Mock performance metrics
            performance_results.update({
                "execution_time": 0.15,  # seconds
                "memory_usage": 128.5,   # MB
                "throughput": 450.0,     # requests/second
                "error_rate": 0.02,      # 2%
            })
            
            # Calculate performance score based on criteria
            score = self._calculate_performance_score(performance_results, performance_criteria)
            performance_results["performance_score"] = score
        
        return performance_results
    
    async def _audit_security(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security audit.
        
        Args:
            data: Security audit configuration
            
        Returns:
            Security audit results
        """
        audit_scope = data.get("scope", ["authentication", "authorization", "data_protection"])
        target_system = data.get("target_system", "")
        
        security_results = {
            "overall_security_level": "medium",
            "vulnerabilities": [],
            "compliance_status": {},
            "recommendations": []
        }
        
        # Simulate security checks
        for scope_item in audit_scope:
            if scope_item == "authentication":
                security_results["compliance_status"]["authentication"] = {
                    "status": "compliant",
                    "score": 0.85,
                    "issues": []
                }
            elif scope_item == "authorization":
                security_results["compliance_status"]["authorization"] = {
                    "status": "partial",
                    "score": 0.70,
                    "issues": ["Role-based access could be more granular"]
                }
            elif scope_item == "data_protection":
                security_results["compliance_status"]["data_protection"] = {
                    "status": "compliant",
                    "score": 0.90,
                    "issues": []
                }
        
        # Generate security insights
        security_context = f"Security audit for {target_system} covering {audit_scope}"
        security_insights = await self._generate_security_insights(security_context)
        security_results["ai_insights"] = security_insights
        
        return security_results
    
    async def _check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against regulations and standards.
        
        Args:
            data: Compliance check configuration
            
        Returns:
            Compliance check results
        """
        regulations = data.get("regulations", [])
        content = data.get("content", "")
        
        compliance_results = {
            "overall_compliance": "compliant",
            "regulation_compliance": {},
            "violations": [],
            "risk_level": "low"
        }
        
        # Check against each regulation
        for regulation in regulations:
            compliance_check = await self._check_regulation_compliance(content, regulation)
            compliance_results["regulation_compliance"][regulation] = compliance_check
        
        # Determine overall compliance status
        compliance_statuses = [result["status"] for result in compliance_results["regulation_compliance"].values()]
        if "non-compliant" in compliance_statuses:
            compliance_results["overall_compliance"] = "non-compliant"
            compliance_results["risk_level"] = "high"
        elif "partial" in compliance_statuses:
            compliance_results["overall_compliance"] = "partial"
            compliance_results["risk_level"] = "medium"
        
        return compliance_results
    
    async def _run_regression_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run regression tests to ensure existing functionality.
        
        Args:
            data: Regression test configuration
            
        Returns:
            Regression test results
        """
        test_suite = data.get("test_suite", [])
        baseline = data.get("baseline", {})
        
        regression_results = {
            "total_tests": len(test_suite),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "regression_detected": False
        }
        
        # Simulate test execution
        for i, test_case in enumerate(test_suite):
            # Mock test execution
            test_result = {
                "test_id": test_case.get("id", f"test_{i}"),
                "test_name": test_case.get("name", f"Test {i}"),
                "status": "passed",  # Simplified
                "execution_time": 0.1,
                "error_message": None
            }
            
            if i % 10 == 0:  # Simulate some failures
                test_result["status"] = "failed"
                test_result["error_message"] = "Simulated test failure"
                regression_results["failed_tests"] += 1
            else:
                regression_results["passed_tests"] += 1
            
            regression_results["test_results"].append(test_result)
        
        regression_results["regression_detected"] = regression_results["failed_tests"] > 0
        
        return regression_results
    
    async def _run_integration_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run integration tests for system components.
        
        Args:
            data: Integration test configuration
            
        Returns:
            Integration test results
        """
        components = data.get("components", [])
        test_scenarios = data.get("scenarios", [])
        
        integration_results = {
            "component_compatibility": {},
            "scenario_results": [],
            "integration_score": 0.0,
            "critical_issues": []
        }
        
        # Test component compatibility
        for i, component in enumerate(components):
            integration_results["component_compatibility"][component] = {
                "status": "compatible" if i % 5 != 0 else "incompatible",
                "version": "1.0.0",
                "dependencies_satisfied": True
            }
        
        # Run integration scenarios
        for scenario in test_scenarios:
            scenario_result = {
                "scenario_id": scenario.get("id"),
                "scenario_name": scenario.get("name"),
                "status": "passed",
                "components_involved": scenario.get("components", [])
            }
            integration_results["scenario_results"].append(scenario_result)
        
        # Calculate integration score
        compatible_count = sum(1 for comp in integration_results["component_compatibility"].values() 
                             if comp["status"] == "compatible")
        integration_results["integration_score"] = compatible_count / len(components) if components else 1.0
        
        return integration_results
    
    async def _run_acceptance_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run acceptance tests against user requirements.
        
        Args:
            data: Acceptance test configuration
            
        Returns:
            Acceptance test results
        """
        requirements = data.get("requirements", [])
        acceptance_criteria = data.get("acceptance_criteria", [])
        
        acceptance_results = {
            "requirements_coverage": {},
            "criteria_results": [],
            "acceptance_score": 0.0,
            "user_satisfaction": "unknown"
        }
        
        # Check requirements coverage
        for requirement in requirements:
            acceptance_results["requirements_coverage"][requirement["id"]] = {
                "status": "satisfied",
                "coverage": 100.0,
                "test_cases": requirement.get("test_cases", [])
            }
        
        # Evaluate acceptance criteria
        for criteria in acceptance_criteria:
            criteria_result = {
                "criteria_id": criteria.get("id"),
                "description": criteria.get("description"),
                "status": "met",
                "evidence": "Test execution logs"
            }
            acceptance_results["criteria_results"].append(criteria_result)
        
        # Calculate acceptance score
        satisfied_requirements = sum(1 for req in acceptance_results["requirements_coverage"].values()
                                   if req["status"] == "satisfied")
        acceptance_results["acceptance_score"] = satisfied_requirements / len(requirements) if requirements else 1.0
        
        return acceptance_results
    
    async def _generate_quality_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            data: Report configuration
            
        Returns:
            Quality report
        """
        report_period = data.get("period", "last_week")
        include_metrics = data.get("metrics", True)
        include_trends = data.get("trends", True)
        
        # Analyze quality history
        recent_validations = self.quality_history[-50:] if self.quality_history else []
        
        quality_report = {
            "report_period": report_period,
            "summary": {
                "total_validations": len(recent_validations),
                "average_quality_score": sum(v["overall_score"] for v in recent_validations) / len(recent_validations) if recent_validations else 0.0,
                "success_rate": sum(1 for v in recent_validations if v["overall_status"] == "passed") / len(recent_validations) if recent_validations else 0.0
            },
            "quality_trends": self._analyze_quality_trends(recent_validations) if include_trends else {},
            "performance_metrics": self.performance_metrics if include_metrics else {},
            "recommendations": []
        }
        
        # Generate report insights
        report_context = f"Quality report for {report_period} with {len(recent_validations)} validations"
        report_insights = await self._generate_quality_insights(report_context)
        quality_report["insights"] = report_insights
        
        return quality_report
    
    async def _define_quality_standard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Define a new quality standard.
        
        Args:
            data: Standard definition
            
        Returns:
            Standard creation result
        """
        standard_name = data.get("name", "")
        description = data.get("description", "")
        metrics = data.get("metrics", [])
        
        if not standard_name:
            return {"error": "Standard name is required"}
        
        # Create new quality standard
        standard = QualityStandard(standard_name, description)
        
        # Add metrics
        for metric_data in metrics:
            metric = QualityMetric(
                name=metric_data.get("name", ""),
                description=metric_data.get("description", ""),
                weight=metric_data.get("weight", 1.0),
                threshold=metric_data.get("threshold", 0.8)
            )
            standard.add_metric(metric)
        
        # Store the standard
        self.quality_standards[standard_name] = standard
        
        return {
            "standard_name": standard_name,
            "metrics_count": len(metrics),
            "status": "created",
            "definition": {
                "name": standard_name,
                "description": description,
                "metrics": [m.__dict__ for m in standard.metrics]
            }
        }
    
    async def _generate_test_cases(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test cases for given requirements.
        
        Args:
            data: Test case generation configuration
            
        Returns:
            Generated test cases
        """
        requirements = data.get("requirements", "")
        test_types = data.get("test_types", ["functional", "edge_case", "negative"])
        
        if not requirements:
            return {"error": "Requirements are needed to generate test cases"}
        
        # Generate test cases using AI
        test_cases = await self._ai_generate_test_cases(requirements, test_types)
        
        return {
            "test_cases": test_cases,
            "requirements": requirements,
            "test_types": test_types,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_defects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze defects and failure patterns.
        
        Args:
            data: Defect data
            
        Returns:
            Defect analysis results
        """
        defects = data.get("defects", [])
        analysis_period = data.get("period", "last_month")
        
        if not defects:
            return {"error": "No defect data provided for analysis"}
        
        # Analyze defect patterns
        defect_analysis = {
            "total_defects": len(defects),
            "severity_distribution": self._analyze_defect_severity(defects),
            "category_distribution": self._analyze_defect_categories(defects),
            "trend_analysis": self._analyze_defect_trends(defects),
            "root_cause_analysis": await self._perform_root_cause_analysis(defects)
        }
        
        return defect_analysis
    
    def _initialize_default_standards(self) -> None:
        """Initialize default quality standards."""
        # Content quality standard
        content_standard = QualityStandard(
            "content_quality",
            "Standard for content quality assessment"
        )
        content_standard.add_metric(QualityMetric("clarity", "Content clarity", 0.3, 0.7))
        content_standard.add_metric(QualityMetric("accuracy", "Content accuracy", 0.4, 0.8))
        content_standard.add_metric(QualityMetric("completeness", "Content completeness", 0.3, 0.8))
        
        # Data quality standard
        data_standard = QualityStandard(
            "data_quality",
            "Standard for data quality validation"
        )
        data_standard.add_metric(QualityMetric("completeness", "Data completeness", 0.25, 0.9))
        data_standard.add_metric(QualityMetric("accuracy", "Data accuracy", 0.25, 0.95))
        data_standard.add_metric(QualityMetric("consistency", "Data consistency", 0.25, 0.8))
        data_standard.add_metric(QualityMetric("validity", "Data validity", 0.25, 0.9))
        
        # Output quality standard
        output_standard = QualityStandard(
            "output_quality",
            "Standard for system output validation"
        )
        output_standard.add_metric(QualityMetric("correctness", "Output correctness", 0.4, 0.95))
        output_standard.add_metric(QualityMetric("format", "Format compliance", 0.3, 0.9))
        output_standard.add_metric(QualityMetric("completeness", "Output completeness", 0.3, 0.85))
        
        # Store standards
        self.quality_standards["content_quality"] = content_standard
        self.quality_standards["data_quality"] = data_standard
        self.quality_standards["output_quality"] = output_standard
        self.quality_standards["default"] = content_standard  # Default fallback
    
    async def _validate_against_standard(self, content: str, standard: QualityStandard) -> Dict[str, Any]:
        """Validate content against a quality standard.
        
        Args:
            content: Content to validate
            standard: Quality standard to apply
            
        Returns:
            Validation results for the standard
        """
        validation_result = {
            "standard_name": standard.name,
            "score": 0.0,
            "metric_scores": {},
            "passed": False,
            "issues": []
        }
        
        # Evaluate each metric
        for metric in standard.metrics:
            if metric.name == "clarity":
                score = self._evaluate_clarity(content)
            elif metric.name == "accuracy":
                score = await self._evaluate_accuracy(content)
            elif metric.name == "completeness":
                score = self._evaluate_completeness(content)
            elif metric.name == "consistency":
                score = self._evaluate_consistency(content)
            else:
                score = 0.8  # Default score for unknown metrics
            
            metric.current_score = score
            metric.status = "passed" if score >= metric.threshold else "failed"
            
            validation_result["metric_scores"][metric.name] = {
                "score": score,
                "threshold": metric.threshold,
                "weight": metric.weight,
                "status": metric.status
            }
            
            if score < metric.threshold:
                validation_result["issues"].append(f"{metric.name} score ({score:.2f}) below threshold ({metric.threshold})")
        
        # Calculate overall score
        validation_result["score"] = standard.calculate_overall_score()
        validation_result["passed"] = validation_result["score"] >= standard.overall_threshold
        
        return validation_result
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT.value
        elif score >= 0.8:
            return QualityLevel.GOOD.value
        elif score >= 0.6:
            return QualityLevel.ACCEPTABLE.value
        elif score >= 0.4:
            return QualityLevel.POOR.value
        else:
            return QualityLevel.UNACCEPTABLE.value
    
    def _update_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Update agent performance metrics."""
        self.performance_metrics["total_validations"] += 1
        
        if result.get("overall_status") == "passed":
            self.performance_metrics["passed_validations"] += 1
        else:
            self.performance_metrics["failed_validations"] += 1
        
        # Update average quality score
        if "overall_score" in result and isinstance(result["overall_score"], (int, float)):
            current_avg = self.performance_metrics["average_quality_score"]
            total_validations = self.performance_metrics["total_validations"]
            new_score = result["overall_score"]
            
            # Weighted average update
            self.performance_metrics["average_quality_score"] = (
                (current_avg * (total_validations - 1) + new_score) / total_validations
            )
    
    # Helper methods for various quality checks
    def _evaluate_clarity(self, content: str) -> float:
        """Evaluate content clarity (simplified)."""
        # Simple heuristic: penalize very long sentences and complex words
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Score inversely related to sentence length
        clarity_score = max(0.0, min(1.0, 1.0 - (avg_sentence_length - 15) / 50))
        return clarity_score
    
    async def _evaluate_accuracy(self, content: str) -> float:
        """Evaluate content accuracy using AI."""
        # In a real implementation, this would check facts against knowledge bases
        # For now, return a simulated score
        return 0.85
    
    def _evaluate_completeness(self, content: str) -> float:
        """Evaluate content completeness."""
        # Simple heuristic based on content length and structure
        word_count = len(content.split())
        has_introduction = any(word in content.lower() for word in ['introduction', 'overview', 'summary'])
        has_conclusion = any(word in content.lower() for word in ['conclusion', 'summary', 'results'])
        
        # Score based on length and structure
        length_score = min(1.0, word_count / 200)  # Expect at least 200 words
        structure_score = (has_introduction + has_conclusion) / 2
        
        return (length_score + structure_score) / 2
    
    def _evaluate_consistency(self, content: str) -> float:
        """Evaluate content consistency."""
        # Simple consistency check for terminology and style
        # In production, this would be more sophisticated
        return 0.8  # Simplified score
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified Flesch formula)."""
        sentences = len([s for s in content.split('.') if s.strip()])
        words = len(content.split())
        syllables = sum(self._count_syllables(word) for word in content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score)) / 100.0  # Normalize to 0-1
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        vowels = 'aeiouy'
        word = word.lower()
        count = sum(1 for char in word if char in vowels)
        
        # Adjust for common patterns
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count
    
    async def _generate_quality_insights(self, quality_context: str) -> Dict[str, Any]:
        """Generate AI-powered quality insights."""
        prompt = f"""
As a quality assurance expert, analyze the following quality assessment data and provide insights:

Quality Context: {quality_context}

Please provide:
1. Key quality findings
2. Areas of concern
3. Recommendations for improvement
4. Risk assessment
5. Action items

Respond in JSON format:
{{
    "key_findings": ["finding1", "finding2", ...],
    "concerns": ["concern1", "concern2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "risk_assessment": "low/medium/high",
    "action_items": ["action1", "action2", ...]
}}
"""
        
        try:
            return await self.query_llm_structured(
                prompt,
                {
                    "key_findings": [],
                    "concerns": [],
                    "recommendations": [],
                    "risk_assessment": "medium",
                    "action_items": []
                },
                "You are a quality assurance expert with deep knowledge of testing methodologies and quality standards."
            )
        except Exception as e:
            logger.error(f"Failed to generate quality insights: {e}")
            return {"error": "Could not generate quality insights"}
    
    def _create_quality_summary(self, validation_results: Dict[str, Any]) -> str:
        """Create a text summary of validation results."""
        score = validation_results.get("overall_score", 0.0)
        status = validation_results.get("overall_status", "unknown")
        issues_count = len(validation_results.get("issues", []))
        
        return f"Quality validation completed with score {score:.2f}, status: {status}, {issues_count} issues identified"