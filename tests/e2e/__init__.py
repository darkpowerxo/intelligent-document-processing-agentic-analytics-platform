"""End-to-end test fixtures and utilities.

This module provides fixtures and utilities for end-to-end tests.
E2E tests test the complete application flow from user perspective.
"""

import pytest
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
from unittest.mock import Mock

from fastapi.testclient import TestClient
from httpx import AsyncClient

from ai_architect_demo.api.main import app


@pytest.fixture(scope="session")
def e2e_test_client():
    """Create a test client for E2E tests."""
    return TestClient(app)


@pytest.fixture
async def e2e_async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for E2E tests."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_documents(temp_dir: Path):
    """Create sample documents for E2E testing."""
    documents = {}
    
    # Sample PDF content (mock)
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
    pdf_file = temp_dir / "sample.pdf"
    pdf_file.write_bytes(pdf_content)
    documents["pdf"] = pdf_file
    
    # Sample text content
    text_content = "This is a sample document for end-to-end testing.\n\nIt contains multiple paragraphs and should be processed by our AI system."
    text_file = temp_dir / "sample.txt"  
    text_file.write_text(text_content)
    documents["text"] = text_file
    
    return documents


@pytest.fixture
def e2e_user_data():
    """Sample user data for E2E tests."""
    return {
        "user_id": "e2e_test_user",
        "username": "testuser",
        "email": "test@example.com",
        "preferences": {
            "notification_enabled": True,
            "analysis_depth": "detailed"
        }
    }


class E2ETestHelpers:
    """Utilities for end-to-end testing."""
    
    @staticmethod
    async def upload_document(client: AsyncClient, file_path: Path, user_id: str = "test_user"):
        """Upload a document through the API."""
        with open(file_path, "rb") as file:
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": (file_path.name, file, "application/octet-stream")},
                data={"user_id": user_id}
            )
        return response
    
    @staticmethod
    async def wait_for_analysis_completion(client: AsyncClient, document_id: str, timeout: int = 60):
        """Wait for document analysis to complete."""
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            response = await client.get(f"/api/v1/documents/{document_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                if status_data.get("status") == "completed":
                    return status_data
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"Analysis did not complete within {timeout} seconds")
    
    @staticmethod
    async def get_analysis_results(client: AsyncClient, document_id: str):
        """Get analysis results for a document."""
        response = await client.get(f"/api/v1/documents/{document_id}/analysis")
        if response.status_code == 200:
            return response.json()
        return None
    
    @staticmethod
    async def trigger_business_analysis(client: AsyncClient, data_params: Dict[str, Any]):
        """Trigger business intelligence analysis."""
        response = await client.post(
            "/api/v1/analytics/business-analysis", 
            json=data_params
        )
        return response
    
    @staticmethod
    async def request_quality_check(client: AsyncClient, subject_id: str, subject_type: str):
        """Request quality assurance check."""
        response = await client.post(
            "/api/v1/quality/check",
            json={
                "subject_id": subject_id,
                "subject_type": subject_type,
                "validation_rules": ["completeness", "accuracy", "consistency"]
            }
        )
        return response
    
    @staticmethod
    def assert_api_success(response, expected_status: int = 200):
        """Assert that an API response indicates success."""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
        return response.json()
    
    @staticmethod
    def assert_analysis_quality(analysis_result: Dict[str, Any]):
        """Assert that analysis results meet quality standards."""
        assert "summary" in analysis_result, "Analysis should include summary"
        assert "key_points" in analysis_result, "Analysis should include key points"
        assert len(analysis_result["summary"]) > 10, "Summary should be meaningful"
        assert len(analysis_result["key_points"]) > 0, "Should extract key points"


@pytest.fixture
def e2e_helpers():
    """Provide E2E test helper utilities.""" 
    return E2ETestHelpers()


# Test data generators
@pytest.fixture
def business_data_sample():
    """Sample business data for analytics testing."""
    return {
        "data_source": "quarterly_reports",
        "time_period": "2024-Q1",
        "metrics": ["revenue", "growth_rate", "customer_satisfaction"],
        "filters": {
            "region": "North America",
            "product_category": "Software"
        }
    }


@pytest.fixture
def quality_check_scenarios():
    """Sample scenarios for quality assurance testing."""
    return [
        {
            "subject_type": "document_analysis",
            "validation_rules": ["completeness", "accuracy"],
            "expected_score_threshold": 0.8
        },
        {
            "subject_type": "business_analysis", 
            "validation_rules": ["data_quality", "statistical_validity"],
            "expected_score_threshold": 0.85
        }
    ]


# Mock external services for E2E tests
@pytest.fixture
def mock_external_services():
    """Mock external services for E2E testing."""
    return {
        "nlp_service": Mock(),
        "ocr_service": Mock(), 
        "notification_service": Mock()
    }