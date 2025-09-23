"""End-to-end tests for complete application workflows.

Tests the entire application from user interaction to final results.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

from httpx import AsyncClient


@pytest.mark.e2e
@pytest.mark.slow
class TestDocumentProcessingWorkflow:
    """Test complete document processing workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_document_analysis_flow(self, 
                                                  e2e_async_client: AsyncClient,
                                                  sample_documents: Dict[str, Path],
                                                  e2e_user_data: Dict[str, Any],
                                                  e2e_helpers):
        """Test complete flow from document upload to analysis completion."""
        
        # 1. Upload document
        pdf_file = sample_documents["pdf"]
        upload_response = await e2e_helpers.upload_document(
            e2e_async_client, 
            pdf_file, 
            e2e_user_data["user_id"]
        )
        
        upload_data = e2e_helpers.assert_api_success(upload_response, 201)
        document_id = upload_data["document_id"]
        
        # 2. Verify document was uploaded
        assert document_id is not None
        assert upload_data["status"] == "uploaded"
        assert upload_data["filename"] == pdf_file.name
        
        # 3. Wait for automatic analysis to complete
        try:
            analysis_status = await e2e_helpers.wait_for_analysis_completion(
                e2e_async_client, 
                document_id, 
                timeout=30
            )
            
            assert analysis_status["status"] == "completed"
            assert analysis_status["document_id"] == document_id
            
        except TimeoutError:
            # If automatic analysis times out, manually trigger it
            trigger_response = await e2e_async_client.post(
                f"/api/v1/documents/{document_id}/analyze"
            )
            e2e_helpers.assert_api_success(trigger_response)
            
            # Wait again
            analysis_status = await e2e_helpers.wait_for_analysis_completion(
                e2e_async_client, 
                document_id, 
                timeout=30
            )
        
        # 4. Get analysis results
        analysis_result = await e2e_helpers.get_analysis_results(
            e2e_async_client, 
            document_id
        )
        
        assert analysis_result is not None
        e2e_helpers.assert_analysis_quality(analysis_result)
        
        # 5. Verify specific analysis components
        assert "document_id" in analysis_result
        assert analysis_result["document_id"] == document_id
        assert "analysis_type" in analysis_result
        assert "confidence_score" in analysis_result
        assert analysis_result["confidence_score"] >= 0.0
        
    @pytest.mark.asyncio
    async def test_multiple_document_batch_processing(self,
                                                    e2e_async_client: AsyncClient,
                                                    sample_documents: Dict[str, Path],
                                                    e2e_user_data: Dict[str, Any],
                                                    e2e_helpers):
        """Test batch processing of multiple documents."""
        
        document_ids = []
        
        # 1. Upload multiple documents
        for doc_type, doc_path in sample_documents.items():
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                doc_path,
                e2e_user_data["user_id"]
            )
            
            upload_data = e2e_helpers.assert_api_success(upload_response, 201)
            document_ids.append(upload_data["document_id"])
        
        # 2. Wait for all analyses to complete
        for document_id in document_ids:
            analysis_status = await e2e_helpers.wait_for_analysis_completion(
                e2e_async_client,
                document_id,
                timeout=45
            )
            assert analysis_status["status"] == "completed"
        
        # 3. Verify all results are available
        for document_id in document_ids:
            analysis_result = await e2e_helpers.get_analysis_results(
                e2e_async_client,
                document_id
            )
            assert analysis_result is not None
            e2e_helpers.assert_analysis_quality(analysis_result)


@pytest.mark.e2e
@pytest.mark.slow
class TestBusinessIntelligenceWorkflow:
    """Test complete business intelligence workflow."""
    
    @pytest.mark.asyncio
    async def test_business_analysis_workflow(self,
                                            e2e_async_client: AsyncClient,
                                            business_data_sample: Dict[str, Any],
                                            e2e_helpers):
        """Test complete business intelligence analysis workflow."""
        
        # 1. Trigger business analysis
        bi_response = await e2e_helpers.trigger_business_analysis(
            e2e_async_client,
            business_data_sample
        )
        
        bi_data = e2e_helpers.assert_api_success(bi_response, 202)
        analysis_id = bi_data["analysis_id"]
        
        # 2. Wait for analysis completion
        await asyncio.sleep(5)  # Give analysis time to process
        
        # 3. Get analysis results
        results_response = await e2e_async_client.get(
            f"/api/v1/analytics/business-analysis/{analysis_id}"
        )
        
        results_data = e2e_helpers.assert_api_success(results_response)
        
        # 4. Verify business analysis results
        assert "insights" in results_data
        assert "recommendations" in results_data
        assert "metrics" in results_data
        assert len(results_data["insights"]) > 0
        assert len(results_data["recommendations"]) > 0
        
        # 5. Verify specific business metrics
        metrics = results_data["metrics"]
        assert isinstance(metrics, dict)
        for metric_name in business_data_sample["metrics"]:
            # Each requested metric should have some value
            assert any(metric_name.lower() in k.lower() for k in metrics.keys())


@pytest.mark.e2e
@pytest.mark.slow
class TestQualityAssuranceWorkflow:
    """Test complete quality assurance workflow."""
    
    @pytest.mark.asyncio
    async def test_quality_check_workflow(self,
                                        e2e_async_client: AsyncClient,
                                        sample_documents: Dict[str, Path],
                                        e2e_user_data: Dict[str, Any],
                                        quality_check_scenarios: list,
                                        e2e_helpers):
        """Test complete quality assurance workflow."""
        
        # 1. First, create some content to quality check
        # Upload and analyze a document
        pdf_file = sample_documents["pdf"]
        upload_response = await e2e_helpers.upload_document(
            e2e_async_client,
            pdf_file,
            e2e_user_data["user_id"]
        )
        
        upload_data = e2e_helpers.assert_api_success(upload_response, 201)
        document_id = upload_data["document_id"]
        
        # Wait for analysis
        await e2e_helpers.wait_for_analysis_completion(
            e2e_async_client,
            document_id,
            timeout=30
        )
        
        # 2. Request quality check on the analysis
        qa_response = await e2e_helpers.request_quality_check(
            e2e_async_client,
            document_id,
            "document_analysis"
        )
        
        qa_data = e2e_helpers.assert_api_success(qa_response, 202)
        check_id = qa_data["check_id"]
        
        # 3. Wait for quality check completion
        await asyncio.sleep(3)
        
        # 4. Get quality check results
        qa_results_response = await e2e_async_client.get(
            f"/api/v1/quality/check/{check_id}"
        )
        
        qa_results = e2e_helpers.assert_api_success(qa_results_response)
        
        # 5. Verify quality check results
        assert "validation_passed" in qa_results
        assert "score" in qa_results
        assert "issues" in qa_results
        assert "recommendations" in qa_results
        
        # Quality score should be reasonable
        assert 0.0 <= qa_results["score"] <= 1.0
        
        # Should have validation results for requested rules
        assert isinstance(qa_results["issues"], list)
        assert isinstance(qa_results["recommendations"], list)


@pytest.mark.e2e
@pytest.mark.slow
class TestIntegratedWorkflow:
    """Test workflows that integrate multiple components."""
    
    @pytest.mark.asyncio
    async def test_document_to_business_intelligence_flow(self,
                                                         e2e_async_client: AsyncClient,
                                                         sample_documents: Dict[str, Path],
                                                         e2e_user_data: Dict[str, Any],
                                                         e2e_helpers):
        """Test flow from document analysis to business intelligence."""
        
        # 1. Upload and analyze document
        text_file = sample_documents["text"]
        upload_response = await e2e_helpers.upload_document(
            e2e_async_client,
            text_file,
            e2e_user_data["user_id"]
        )
        
        upload_data = e2e_helpers.assert_api_success(upload_response, 201)
        document_id = upload_data["document_id"]
        
        # 2. Wait for document analysis
        await e2e_helpers.wait_for_analysis_completion(
            e2e_async_client,
            document_id,
            timeout=30
        )
        
        # 3. Use document analysis results for business intelligence
        bi_params = {
            "data_source": "document_analysis",
            "source_document_id": document_id,
            "analysis_type": "content_insights",
            "metrics": ["sentiment", "key_topics", "action_items"]
        }
        
        bi_response = await e2e_helpers.trigger_business_analysis(
            e2e_async_client,
            bi_params
        )
        
        bi_data = e2e_helpers.assert_api_success(bi_response, 202)
        analysis_id = bi_data["analysis_id"]
        
        # 4. Get integrated analysis results
        await asyncio.sleep(5)
        
        results_response = await e2e_async_client.get(
            f"/api/v1/analytics/business-analysis/{analysis_id}"
        )
        
        results_data = e2e_helpers.assert_api_success(results_response)
        
        # 5. Verify integrated results reference original document
        assert "source_document_id" in results_data
        assert results_data["source_document_id"] == document_id
        assert "insights" in results_data
        assert len(results_data["insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_validation_flow(self,
                                                     e2e_async_client: AsyncClient,
                                                     sample_documents: Dict[str, Path],
                                                     e2e_user_data: Dict[str, Any],
                                                     e2e_helpers):
        """Test complete end-to-end flow with quality validation at each step."""
        
        # 1. Document upload with validation
        pdf_file = sample_documents["pdf"]
        upload_response = await e2e_helpers.upload_document(
            e2e_async_client,
            pdf_file,
            e2e_user_data["user_id"]
        )
        
        upload_data = e2e_helpers.assert_api_success(upload_response, 201)
        document_id = upload_data["document_id"]
        
        # 2. Document analysis with validation
        await e2e_helpers.wait_for_analysis_completion(
            e2e_async_client,
            document_id,
            timeout=30
        )
        
        # Validate document analysis
        doc_qa_response = await e2e_helpers.request_quality_check(
            e2e_async_client,
            document_id,
            "document_analysis"
        )
        
        doc_qa_data = e2e_helpers.assert_api_success(doc_qa_response, 202)
        
        # 3. Business intelligence analysis
        bi_params = {
            "data_source": "document_analysis",
            "source_document_id": document_id,
            "metrics": ["key_insights", "trends"]
        }
        
        bi_response = await e2e_helpers.trigger_business_analysis(
            e2e_async_client,
            bi_params
        )
        
        bi_data = e2e_helpers.assert_api_success(bi_response, 202)
        bi_analysis_id = bi_data["analysis_id"]
        
        await asyncio.sleep(5)  # Wait for BI analysis
        
        # Validate business analysis
        bi_qa_response = await e2e_helpers.request_quality_check(
            e2e_async_client,
            bi_analysis_id,
            "business_analysis"
        )
        
        bi_qa_data = e2e_helpers.assert_api_success(bi_qa_response, 202)
        
        # 4. Verify all quality checks
        await asyncio.sleep(3)  # Wait for QA completion
        
        # Check document QA results
        doc_qa_results = await e2e_async_client.get(
            f"/api/v1/quality/check/{doc_qa_data['check_id']}"
        )
        doc_qa_data = e2e_helpers.assert_api_success(doc_qa_results)
        
        # Check BI QA results  
        bi_qa_results = await e2e_async_client.get(
            f"/api/v1/quality/check/{bi_qa_data['check_id']}"
        )
        bi_qa_data = e2e_helpers.assert_api_success(bi_qa_results)
        
        # Both should pass basic quality checks
        assert doc_qa_data["validation_passed"] is True or doc_qa_data["score"] > 0.6
        assert bi_qa_data["validation_passed"] is True or bi_qa_data["score"] > 0.6