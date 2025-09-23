"""Document Analyzer Agent for intelligent document processing and analysis.

This agent specializes in analyzing various document formats, extracting insights,
and providing structured analysis results using local LLMs.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from ai_architect_demo.core.logging import get_logger, log_function_call
from ai_architect_demo.agents.base_agent import (
    BaseAgent, AgentTask, AgentRole, AgentCapabilities
)
from ai_architect_demo.data.document_processor import DocumentProcessor

logger = get_logger(__name__)


class DocumentAnalyzerAgent(BaseAgent):
    """Specialized agent for document analysis and processing."""
    
    def __init__(
        self,
        agent_id: str = "doc_analyzer_001",
        name: str = "Document Analyzer",
        description: str = "Analyzes documents and extracts structured insights",
        ollama_endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1:latest"
    ):
        """Initialize the Document Analyzer Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable agent name
            description: Description of agent's purpose
            ollama_endpoint: Ollama server endpoint
            model_name: LLM model to use
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.DOCUMENT_ANALYZER,
            name=name,
            description=description,
            ollama_endpoint=ollama_endpoint,
            model_name=model_name
        )
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        # Analysis templates and prompts
        self.analysis_prompts = {
            "summary": self._get_summary_prompt(),
            "key_points": self._get_key_points_prompt(),
            "sentiment": self._get_sentiment_prompt(),
            "entities": self._get_entities_prompt(),
            "classification": self._get_classification_prompt(),
            "quality_assessment": self._get_quality_prompt(),
            "action_items": self._get_action_items_prompt()
        }
    
    def _define_capabilities(self) -> AgentCapabilities:
        """Define the document analyzer's capabilities."""
        return AgentCapabilities(
            supported_tasks=[
                "document_analysis",
                "text_summarization", 
                "entity_extraction",
                "sentiment_analysis",
                "document_classification",
                "content_quality_assessment",
                "action_item_extraction",
                "key_phrase_extraction",
                "document_comparison",
                "compliance_check"
            ],
            max_concurrent_tasks=3,
            specializations=[
                "Natural Language Processing",
                "Information Extraction", 
                "Document Understanding",
                "Content Analysis",
                "Text Mining"
            ],
            required_resources=[
                "local_llm",
                "document_processor"
            ]
        )
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a document analysis task.
        
        Args:
            task: Task to process
            
        Returns:
            Analysis results
        """
        log_function_call("process_task", agent_id=self.agent_id, task_type=task.task_type)
        
        task_handlers = {
            "document_analysis": self._analyze_document,
            "text_summarization": self._summarize_text,
            "entity_extraction": self._extract_entities,
            "sentiment_analysis": self._analyze_sentiment,
            "document_classification": self._classify_document,
            "content_quality_assessment": self._assess_quality,
            "action_item_extraction": self._extract_action_items,
            "key_phrase_extraction": self._extract_key_phrases,
            "document_comparison": self._compare_documents,
            "compliance_check": self._check_compliance
        }
        
        handler = task_handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"Unsupported task type: {task.task_type}")
        
        return await handler(task.data)
    
    async def _analyze_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive document analysis.
        
        Args:
            data: Task data containing document information
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting comprehensive document analysis")
        
        # Extract document content
        content = await self._extract_document_content(data)
        
        if not content:
            return {"error": "Could not extract document content"}
        
        # Perform multiple analysis types
        results = {
            "document_info": {
                "content_length": len(content),
                "word_count": len(content.split()),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        # Run analysis tasks in parallel where possible
        analysis_tasks = []
        
        # Summary analysis
        summary_result = await self._perform_llm_analysis(
            content, 
            "summary",
            {"max_length": 200, "style": "professional", "focus_areas": []}
        )
        results["summary"] = summary_result
        
        # Key points extraction
        key_points_result = await self._perform_llm_analysis(
            content,
            "key_points", 
            {"max_points": 10}
        )
        results["key_points"] = key_points_result
        
        # Entity extraction
        entities_result = await self._perform_llm_analysis(
            content,
            "entities",
            {"types": ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "PRODUCT"]}
        )
        results["entities"] = entities_result
        
        # Sentiment analysis
        sentiment_result = await self._perform_llm_analysis(
            content,
            "sentiment",
            {"aspects": ["overall", "specific_topics"]}
        )
        results["sentiment"] = sentiment_result
        
        # Document classification
        classification_result = await self._perform_llm_analysis(
            content,
            "classification",
            {"categories": data.get("classification_categories", [])}
        )
        results["classification"] = classification_result
        
        # Quality assessment
        quality_result = await self._perform_llm_analysis(
            content,
            "quality_assessment",
            {"criteria": ["clarity", "completeness", "accuracy", "structure"]}
        )
        results["quality"] = quality_result
        
        logger.info("Comprehensive document analysis completed")
        return results
    
    async def _summarize_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text summary.
        
        Args:
            data: Task data
            
        Returns:
            Summary results
        """
        content = await self._extract_document_content(data)
        
        summary_config = {
            "max_length": data.get("max_length", 200),
            "style": data.get("style", "professional"),
            "focus_areas": data.get("focus_areas", [])
        }
        
        result = await self._perform_llm_analysis(content, "summary", summary_config)
        
        return {
            "summary": result.get("summary", ""),
            "key_themes": result.get("key_themes", []),
            "summary_length": len(result.get("summary", "")),
            "original_length": len(content),
            "compression_ratio": len(result.get("summary", "")) / len(content) if content else 0
        }
    
    async def _extract_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from text.
        
        Args:
            data: Task data
            
        Returns:
            Extracted entities
        """
        content = await self._extract_document_content(data)
        
        entity_config = {
            "types": data.get("entity_types", ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]),
            "include_confidence": data.get("include_confidence", True)
        }
        
        result = await self._perform_llm_analysis(content, "entities", entity_config)
        
        return {
            "entities": result.get("entities", {}),
            "entity_count": sum(len(entities) for entities in result.get("entities", {}).values()),
            "entity_density": sum(len(entities) for entities in result.get("entities", {}).values()) / len(content.split()) if content else 0
        }
    
    async def _analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document sentiment.
        
        Args:
            data: Task data
            
        Returns:
            Sentiment analysis results
        """
        content = await self._extract_document_content(data)
        
        sentiment_config = {
            "aspects": data.get("aspects", ["overall"]),
            "granularity": data.get("granularity", "document")  # document, paragraph, sentence
        }
        
        result = await self._perform_llm_analysis(content, "sentiment", sentiment_config)
        
        return {
            "overall_sentiment": result.get("overall_sentiment", {}),
            "aspect_sentiments": result.get("aspect_sentiments", {}),
            "confidence_score": result.get("confidence_score", 0.0),
            "emotional_indicators": result.get("emotional_indicators", [])
        }
    
    async def _classify_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document into categories.
        
        Args:
            data: Task data
            
        Returns:
            Classification results
        """
        content = await self._extract_document_content(data)
        
        classification_config = {
            "categories": data.get("categories", []),
            "multi_label": data.get("multi_label", False),
            "confidence_threshold": data.get("confidence_threshold", 0.5)
        }
        
        result = await self._perform_llm_analysis(content, "classification", classification_config)
        
        return {
            "primary_category": result.get("primary_category", ""),
            "all_categories": result.get("all_categories", []),
            "confidence_scores": result.get("confidence_scores", {}),
            "reasoning": result.get("reasoning", "")
        }
    
    async def _assess_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess document quality.
        
        Args:
            data: Task data
            
        Returns:
            Quality assessment results
        """
        content = await self._extract_document_content(data)
        
        quality_config = {
            "criteria": data.get("criteria", ["clarity", "completeness", "accuracy", "structure"]),
            "scoring_scale": data.get("scoring_scale", "1-10")
        }
        
        result = await self._perform_llm_analysis(content, "quality_assessment", quality_config)
        
        return {
            "overall_score": result.get("overall_score", 0),
            "criterion_scores": result.get("criterion_scores", {}),
            "strengths": result.get("strengths", []),
            "improvement_areas": result.get("improvement_areas", []),
            "recommendations": result.get("recommendations", [])
        }
    
    async def _extract_action_items(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action items from document.
        
        Args:
            data: Task data
            
        Returns:
            Action items
        """
        content = await self._extract_document_content(data)
        
        action_config = {
            "priority_levels": data.get("priority_levels", ["high", "medium", "low"]),
            "include_deadlines": data.get("include_deadlines", True),
            "include_assignees": data.get("include_assignees", True)
        }
        
        result = await self._perform_llm_analysis(content, "action_items", action_config)
        
        return {
            "action_items": result.get("action_items", []),
            "total_actions": len(result.get("action_items", [])),
            "priority_distribution": result.get("priority_distribution", {}),
            "urgent_actions": result.get("urgent_actions", [])
        }
    
    async def _extract_key_phrases(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key phrases from document.
        
        Args:
            data: Task data
            
        Returns:
            Key phrases
        """
        content = await self._extract_document_content(data)
        
        # Use a simplified approach for key phrase extraction
        prompt = f"""
Analyze the following text and extract the most important key phrases and terms.

Text:
{content[:3000]}  # Limit content for processing

Please identify:
1. The 10 most important key phrases
2. Technical terms and jargon
3. Topic keywords
4. Important concepts

Respond in JSON format:
{{
    "key_phrases": ["phrase1", "phrase2", ...],
    "technical_terms": ["term1", "term2", ...],
    "topic_keywords": ["keyword1", "keyword2", ...],
    "concepts": ["concept1", "concept2", ...]
}}
"""
        
        result = await self.query_llm_structured(
            prompt,
            {
                "key_phrases": [],
                "technical_terms": [],
                "topic_keywords": [],
                "concepts": []
            },
            "You are an expert in text analysis and information extraction."
        )
        
        return {
            **result,
            "phrase_count": len(result.get("key_phrases", [])),
            "term_density": len(result.get("technical_terms", [])) / len(content.split()) if content else 0
        }
    
    async def _compare_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple documents.
        
        Args:
            data: Task data with documents to compare
            
        Returns:
            Comparison results
        """
        documents = data.get("documents", [])
        
        if len(documents) < 2:
            return {"error": "Need at least 2 documents for comparison"}
        
        # Extract content from all documents
        contents = []
        for doc_data in documents:
            content = await self._extract_document_content(doc_data)
            contents.append(content)
        
        # Perform comparison analysis
        prompt = f"""
Compare the following documents and provide a detailed analysis:

Document 1:
{contents[0][:2000] if contents[0] else "No content"}

Document 2:
{contents[1][:2000] if contents[1] else "No content"}

Please provide:
1. Similarities between the documents
2. Key differences
3. Unique content in each document
4. Overall similarity score (0-100)
5. Recommendations based on the comparison

Respond in JSON format:
{{
    "similarities": ["similarity1", "similarity2", ...],
    "differences": ["diff1", "diff2", ...],
    "unique_content": {{
        "document1": ["unique1", "unique2", ...],
        "document2": ["unique1", "unique2", ...]
    }},
    "similarity_score": 0-100,
    "recommendations": ["rec1", "rec2", ...]
}}
"""
        
        result = await self.query_llm_structured(
            prompt,
            {
                "similarities": [],
                "differences": [],
                "unique_content": {"document1": [], "document2": []},
                "similarity_score": 0,
                "recommendations": []
            },
            "You are an expert document analyst specializing in comparative analysis."
        )
        
        return {
            **result,
            "documents_compared": len(documents),
            "comparison_timestamp": datetime.now().isoformat()
        }
    
    async def _check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check document compliance against standards.
        
        Args:
            data: Task data with compliance requirements
            
        Returns:
            Compliance check results
        """
        content = await self._extract_document_content(data)
        compliance_standards = data.get("standards", [])
        
        if not compliance_standards:
            return {"error": "No compliance standards specified"}
        
        prompt = f"""
Review the following document for compliance with these standards:
Standards: {', '.join(compliance_standards)}

Document:
{content[:3000]}

Please assess:
1. Compliance status for each standard
2. Areas of non-compliance
3. Risk level (low/medium/high)
4. Recommendations for improvement

Respond in JSON format:
{{
    "overall_compliance": "compliant/non-compliant/partial",
    "standard_compliance": {{
        "standard1": {{"status": "compliant/non-compliant", "score": 0-100}},
        "standard2": {{"status": "compliant/non-compliant", "score": 0-100}}
    }},
    "violations": ["violation1", "violation2", ...],
    "risk_level": "low/medium/high",
    "recommendations": ["rec1", "rec2", ...]
}}
"""
        
        result = await self.query_llm_structured(
            prompt,
            {
                "overall_compliance": "unknown",
                "standard_compliance": {},
                "violations": [],
                "risk_level": "medium",
                "recommendations": []
            },
            "You are a compliance expert specializing in document review and regulatory standards."
        )
        
        return {
            **result,
            "standards_checked": compliance_standards,
            "compliance_timestamp": datetime.now().isoformat()
        }
    
    async def _extract_document_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from document data.
        
        Args:
            data: Document data
            
        Returns:
            Extracted text content
        """
        # Handle different input formats
        if "content" in data:
            return str(data["content"])
        elif "file_path" in data:
            try:
                return await self.document_processor.process_document(data["file_path"])
            except Exception as e:
                logger.error(f"Failed to process document from file: {e}")
                return ""
        elif "text" in data:
            return str(data["text"])
        else:
            logger.warning("No recognizable content format in task data")
            return ""
    
    async def _perform_llm_analysis(
        self,
        content: str,
        analysis_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform LLM-based analysis on content.
        
        Args:
            content: Text content to analyze
            analysis_type: Type of analysis to perform
            config: Analysis configuration
            
        Returns:
            Analysis results
        """
        prompt_template = self.analysis_prompts.get(analysis_type)
        if not prompt_template:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Format prompt with content and config
        prompt = prompt_template.format(content=content[:4000], **config)  # Limit content length
        
        # Get appropriate system prompt
        system_prompt = self._get_system_prompt(analysis_type)
        
        try:
            # For structured output, try to get JSON response
            if analysis_type in ["entities", "classification", "quality_assessment"]:
                return await self.query_llm_structured(
                    prompt,
                    self._get_response_schema(analysis_type),
                    system_prompt
                )
            else:
                # For less structured output, parse manually
                response = await self.query_llm(prompt, system_prompt)
                return self._parse_analysis_response(response, analysis_type)
                
        except Exception as e:
            logger.error(f"LLM analysis failed for {analysis_type}: {e}")
            return {"error": f"Analysis failed: {e}"}
    
    def _get_summary_prompt(self) -> str:
        """Get prompt template for summarization."""
        return """
Please provide a comprehensive summary of the following text.

Text:
{content}

Requirements:
- Maximum length: {max_length} words
- Style: {style}
- Focus areas: {focus_areas}

Please also identify the key themes present in the text.

Provide your response as:
Summary: [your summary here]
Key Themes: [list of key themes]
"""
    
    def _get_key_points_prompt(self) -> str:
        """Get prompt template for key points extraction."""
        return """
Extract the {max_points} most important key points from the following text:

{content}

Present the key points as a numbered list, ordered by importance.
"""
    
    def _get_sentiment_prompt(self) -> str:
        """Get prompt template for sentiment analysis."""
        return """
Analyze the sentiment of the following text:

{content}

Please provide:
1. Overall sentiment (positive/negative/neutral) with confidence score
2. Sentiment for specific aspects: {aspects}
3. Any emotional indicators present

Format as:
Overall: [sentiment] (confidence: X%)
Aspects: [analysis for each aspect]
Emotional Indicators: [list of indicators]
"""
    
    def _get_entities_prompt(self) -> str:
        """Get prompt template for entity extraction."""
        return """
Extract named entities from the following text:

{content}

Entity types to extract: {types}

Respond in JSON format:
{{
    "entities": {{
        "PERSON": ["entity1", "entity2"],
        "ORGANIZATION": ["entity1", "entity2"],
        ...
    }}
}}
"""
    
    def _get_classification_prompt(self) -> str:
        """Get prompt template for classification."""
        return """
Classify the following document into appropriate categories:

{content}

Available categories: {categories}

Respond in JSON format:
{{
    "primary_category": "category_name",
    "all_categories": ["cat1", "cat2"],
    "confidence_scores": {{"cat1": 0.8, "cat2": 0.6}},
    "reasoning": "explanation of classification"
}}
"""
    
    def _get_quality_prompt(self) -> str:
        """Get prompt template for quality assessment."""
        return """
Assess the quality of the following document based on these criteria: {criteria}

{content}

Rate each criterion on a scale of 1-10 and provide overall assessment.

Respond in JSON format:
{{
    "overall_score": 8,
    "criterion_scores": {{"clarity": 8, "completeness": 7}},
    "strengths": ["strength1", "strength2"],
    "improvement_areas": ["area1", "area2"],
    "recommendations": ["rec1", "rec2"]
}}
"""
    
    def _get_action_items_prompt(self) -> str:
        """Get prompt template for action item extraction."""
        return """
Extract action items from the following text:

{content}

Look for:
- Tasks to be completed
- Assignments and responsibilities
- Deadlines and timeframes
- Priority levels: {priority_levels}

Format each action item with priority, description, and any mentioned deadlines or assignees.
"""
    
    def _get_system_prompt(self, analysis_type: str) -> str:
        """Get system prompt for specific analysis type."""
        prompts = {
            "summary": "You are an expert at creating clear, concise summaries that capture the essence of documents.",
            "key_points": "You are skilled at identifying and prioritizing the most important information in documents.",
            "sentiment": "You are an expert in sentiment analysis and emotional intelligence in text.",
            "entities": "You are a named entity recognition specialist with expertise in information extraction.",
            "classification": "You are a document classification expert with knowledge across multiple domains.",
            "quality_assessment": "You are a document quality expert who evaluates clarity, completeness, and effectiveness.",
            "action_items": "You are a project management expert skilled at identifying actionable tasks and responsibilities."
        }
        return prompts.get(analysis_type, "You are a helpful AI assistant specializing in document analysis.")
    
    def _get_response_schema(self, analysis_type: str) -> Dict[str, Any]:
        """Get expected response schema for structured analysis."""
        schemas = {
            "entities": {
                "entities": {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": []}
            },
            "classification": {
                "primary_category": "",
                "all_categories": [],
                "confidence_scores": {},
                "reasoning": ""
            },
            "quality_assessment": {
                "overall_score": 0,
                "criterion_scores": {},
                "strengths": [],
                "improvement_areas": [],
                "recommendations": []
            }
        }
        return schemas.get(analysis_type, {})
    
    def _parse_analysis_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse unstructured LLM response into structured format.
        
        Args:
            response: Raw LLM response
            analysis_type: Type of analysis
            
        Returns:
            Structured response data
        """
        # This is a simplified parser - in production you'd want more robust parsing
        if analysis_type == "summary":
            lines = response.strip().split('\n')
            summary = ""
            key_themes = []
            
            for line in lines:
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Key Themes:"):
                    themes_text = line.replace("Key Themes:", "").strip()
                    key_themes = [theme.strip() for theme in themes_text.split(',')]
            
            return {
                "summary": summary,
                "key_themes": key_themes
            }
        
        # Default fallback
        return {"response": response}