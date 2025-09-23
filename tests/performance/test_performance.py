"""Performance and load testing for the AI architecture."""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
from typing import List, Dict, Any
from pathlib import Path

from httpx import AsyncClient


@pytest.mark.performance 
@pytest.mark.slow
class TestPerformanceBaselines:
    """Establish performance baselines for key operations."""
    
    @pytest.mark.asyncio
    async def test_document_upload_performance(self,
                                             e2e_async_client: AsyncClient,
                                             sample_documents: Dict[str, Path],
                                             e2e_user_data: Dict[str, Any],
                                             e2e_helpers):
        """Test document upload performance."""
        
        upload_times = []
        document_sizes = []
        
        # Test with different document types
        for doc_type, doc_path in sample_documents.items():
            file_size = doc_path.stat().st_size
            document_sizes.append(file_size)
            
            # Measure upload time
            start_time = time.perf_counter()
            
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                doc_path,
                e2e_user_data["user_id"]
            )
            
            end_time = time.perf_counter()
            upload_time = end_time - start_time
            upload_times.append(upload_time)
            
            e2e_helpers.assert_api_success(upload_response, 201)
        
        # Performance assertions
        avg_upload_time = mean(upload_times)
        max_upload_time = max(upload_times)
        
        # Upload should generally complete within reasonable time
        assert avg_upload_time < 5.0, f"Average upload time {avg_upload_time:.2f}s too slow"
        assert max_upload_time < 10.0, f"Max upload time {max_upload_time:.2f}s too slow"
        
        # Performance metrics for reporting
        print(f"\nUpload Performance:")
        print(f"  Average time: {avg_upload_time:.2f}s")
        print(f"  Max time: {max_upload_time:.2f}s")
        print(f"  Document sizes: {[f'{s/1024:.1f}KB' for s in document_sizes]}")
    
    @pytest.mark.asyncio
    async def test_document_analysis_performance(self,
                                               e2e_async_client: AsyncClient,
                                               sample_documents: Dict[str, Path],
                                               e2e_user_data: Dict[str, Any],
                                               e2e_helpers):
        """Test document analysis performance."""
        
        analysis_times = []
        document_ids = []
        
        # First upload documents
        for doc_type, doc_path in sample_documents.items():
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                doc_path,
                e2e_user_data["user_id"]
            )
            
            upload_data = e2e_helpers.assert_api_success(upload_response, 201)
            document_ids.append(upload_data["document_id"])
        
        # Measure analysis times
        for document_id in document_ids:
            start_time = time.perf_counter()
            
            # Wait for analysis to complete
            await e2e_helpers.wait_for_analysis_completion(
                e2e_async_client,
                document_id,
                timeout=60
            )
            
            end_time = time.perf_counter()
            analysis_time = end_time - start_time
            analysis_times.append(analysis_time)
        
        # Performance assertions
        avg_analysis_time = mean(analysis_times)
        max_analysis_time = max(analysis_times)
        
        # Analysis should complete within reasonable time
        assert avg_analysis_time < 30.0, f"Average analysis time {avg_analysis_time:.2f}s too slow"
        assert max_analysis_time < 60.0, f"Max analysis time {max_analysis_time:.2f}s too slow"
        
        print(f"\nAnalysis Performance:")
        print(f"  Average time: {avg_analysis_time:.2f}s")
        print(f"  Max time: {max_analysis_time:.2f}s")
        print(f"  Median time: {median(analysis_times):.2f}s")
    
    @pytest.mark.asyncio
    async def test_api_response_time_performance(self,
                                               e2e_async_client: AsyncClient,
                                               e2e_helpers):
        """Test API endpoint response times."""
        
        endpoints = [
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/status"),
            ("GET", "/api/v1/analytics/metrics"),
        ]
        
        response_times = {}
        
        for method, endpoint in endpoints:
            times = []
            
            # Test each endpoint multiple times
            for _ in range(10):
                start_time = time.perf_counter()
                
                if method == "GET":
                    response = await e2e_async_client.get(endpoint)
                
                end_time = time.perf_counter()
                response_time = end_time - start_time
                times.append(response_time)
                
                # Basic success assertion
                assert response.status_code in [200, 404]  # 404 acceptable for some endpoints
            
            response_times[endpoint] = {
                "avg": mean(times),
                "max": max(times),
                "min": min(times),
                "median": median(times)
            }
        
        # Performance assertions
        for endpoint, metrics in response_times.items():
            assert metrics["avg"] < 1.0, f"{endpoint} average response time {metrics['avg']:.3f}s too slow"
            assert metrics["max"] < 2.0, f"{endpoint} max response time {metrics['max']:.3f}s too slow"
        
        print(f"\nAPI Response Times:")
        for endpoint, metrics in response_times.items():
            print(f"  {endpoint}: avg={metrics['avg']:.3f}s, max={metrics['max']:.3f}s")


@pytest.mark.load
@pytest.mark.slow
class TestLoadHandling:
    """Test system behavior under load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_document_uploads(self,
                                             e2e_async_client: AsyncClient,
                                             sample_documents: Dict[str, Path],
                                             e2e_user_data: Dict[str, Any],
                                             e2e_helpers):
        """Test concurrent document upload handling."""
        
        concurrent_uploads = 5
        text_file = sample_documents["text"]
        
        async def upload_document():
            """Upload a single document."""
            return await e2e_helpers.upload_document(
                e2e_async_client,
                text_file,
                e2e_user_data["user_id"]
            )
        
        # Launch concurrent uploads
        start_time = time.perf_counter()
        
        tasks = [upload_document() for _ in range(concurrent_uploads)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Verify all uploads succeeded
        successful_uploads = 0
        for response in responses:
            if not isinstance(response, Exception):
                if response.status_code == 201:
                    successful_uploads += 1
        
        # At least 80% should succeed under load
        success_rate = successful_uploads / concurrent_uploads
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} too low under load"
        
        # Total time should be reasonable for concurrent operations
        expected_max_time = 15.0  # Allow some overhead for concurrency
        assert total_time < expected_max_time, f"Concurrent uploads took {total_time:.2f}s"
        
        print(f"\nConcurrent Upload Results:")
        print(f"  Successful uploads: {successful_uploads}/{concurrent_uploads}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self,
                                              e2e_async_client: AsyncClient,
                                              sample_documents: Dict[str, Path],
                                              e2e_user_data: Dict[str, Any],
                                              e2e_helpers):
        """Test concurrent analysis request handling."""
        
        # First, upload documents for analysis
        document_ids = []
        for doc_type, doc_path in list(sample_documents.items())[:3]:  # Limit to 3 docs
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                doc_path,
                e2e_user_data["user_id"]
            )
            
            upload_data = e2e_helpers.assert_api_success(upload_response, 201)
            document_ids.append(upload_data["document_id"])
        
        # Now trigger concurrent analyses
        async def analyze_document(document_id):
            """Analyze a single document."""
            try:
                return await e2e_helpers.wait_for_analysis_completion(
                    e2e_async_client,
                    document_id,
                    timeout=90  # Longer timeout for concurrent load
                )
            except Exception as e:
                return e
        
        start_time = time.perf_counter()
        
        tasks = [analyze_document(doc_id) for doc_id in document_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Count successful analyses
        successful_analyses = 0
        for result in results:
            if not isinstance(result, Exception):
                if isinstance(result, dict) and result.get("status") == "completed":
                    successful_analyses += 1
        
        # At least 2/3 should succeed under concurrent load
        success_rate = successful_analyses / len(document_ids)
        assert success_rate >= 0.67, f"Analysis success rate {success_rate:.2f} too low"
        
        print(f"\nConcurrent Analysis Results:")
        print(f"  Successful analyses: {successful_analyses}/{len(document_ids)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.2f}s")


@pytest.mark.stress
@pytest.mark.slow
class TestStressConditions:
    """Test system behavior under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self,
                                         e2e_async_client: AsyncClient,
                                         sample_documents: Dict[str, Path],
                                         e2e_user_data: Dict[str, Any],
                                         e2e_helpers):
        """Test memory usage under document processing load."""
        
        # Use larger document for memory stress testing
        largest_doc = max(sample_documents.values(), key=lambda p: p.stat().st_size)
        
        document_ids = []
        
        # Upload multiple copies of the largest document
        for i in range(10):
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                largest_doc,
                e2e_user_data["user_id"]
            )
            
            upload_data = e2e_helpers.assert_api_success(upload_response, 201)
            document_ids.append(upload_data["document_id"])
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.5)
        
        print(f"\nMemory Stress Test:")
        print(f"  Uploaded {len(document_ids)} documents")
        print(f"  Document size: {largest_doc.stat().st_size / 1024:.1f}KB")
        
        # Test that the system still responds
        health_response = await e2e_async_client.get("/api/v1/health")
        assert health_response.status_code in [200, 404]  # Should still respond
        
    @pytest.mark.asyncio  
    async def test_rapid_request_handling(self,
                                        e2e_async_client: AsyncClient,
                                        e2e_helpers):
        """Test handling of rapid successive requests."""
        
        # Make rapid requests to health endpoint
        request_count = 50
        start_time = time.perf_counter()
        
        async def make_request():
            return await e2e_async_client.get("/api/v1/health")
        
        tasks = [make_request() for _ in range(request_count)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Count successful responses
        successful_requests = sum(
            1 for r in responses 
            if not isinstance(r, Exception) and r.status_code in [200, 404]
        )
        
        success_rate = successful_requests / request_count
        requests_per_second = request_count / total_time
        
        # System should handle rapid requests reasonably well
        assert success_rate >= 0.9, f"Success rate {success_rate:.2f} under rapid requests"
        assert requests_per_second >= 10, f"Request rate {requests_per_second:.1f} req/s too slow"
        
        print(f"\nRapid Request Test:")
        print(f"  Successful requests: {successful_requests}/{request_count}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Requests per second: {requests_per_second:.1f}")
        print(f"  Total time: {total_time:.2f}s")


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Benchmark tests for performance regression detection."""
    
    @pytest.mark.asyncio
    async def test_document_processing_benchmark(self,
                                               e2e_async_client: AsyncClient,
                                               sample_documents: Dict[str, Path],
                                               e2e_user_data: Dict[str, Any],
                                               e2e_helpers):
        """Comprehensive benchmark of document processing pipeline."""
        
        benchmark_results = {
            "upload_times": [],
            "analysis_times": [],
            "total_pipeline_times": []
        }
        
        # Run benchmark on each document type
        for doc_type, doc_path in sample_documents.items():
            pipeline_start = time.perf_counter()
            
            # 1. Upload phase
            upload_start = time.perf_counter()
            upload_response = await e2e_helpers.upload_document(
                e2e_async_client,
                doc_path,
                e2e_user_data["user_id"]
            )
            upload_end = time.perf_counter()
            upload_time = upload_end - upload_start
            
            upload_data = e2e_helpers.assert_api_success(upload_response, 201)
            document_id = upload_data["document_id"]
            
            # 2. Analysis phase
            analysis_start = time.perf_counter()
            await e2e_helpers.wait_for_analysis_completion(
                e2e_async_client,
                document_id,
                timeout=60
            )
            analysis_end = time.perf_counter()
            analysis_time = analysis_end - analysis_start
            
            pipeline_end = time.perf_counter()
            total_time = pipeline_end - pipeline_start
            
            # Record results
            benchmark_results["upload_times"].append(upload_time)
            benchmark_results["analysis_times"].append(analysis_time)
            benchmark_results["total_pipeline_times"].append(total_time)
        
        # Calculate benchmark statistics
        stats = {}
        for phase, times in benchmark_results.items():
            stats[phase] = {
                "avg": mean(times),
                "median": median(times),
                "min": min(times),
                "max": max(times),
                "std": stdev(times) if len(times) > 1 else 0
            }
        
        # Performance regression checks (adjust thresholds as needed)
        assert stats["upload_times"]["avg"] < 3.0, "Upload performance regression detected"
        assert stats["analysis_times"]["avg"] < 25.0, "Analysis performance regression detected"  
        assert stats["total_pipeline_times"]["avg"] < 30.0, "Pipeline performance regression detected"
        
        # Print benchmark results
        print(f"\nPerformance Benchmark Results:")
        for phase, phase_stats in stats.items():
            print(f"  {phase}:")
            print(f"    Average: {phase_stats['avg']:.2f}s")
            print(f"    Median:  {phase_stats['median']:.2f}s")
            print(f"    Min:     {phase_stats['min']:.2f}s")  
            print(f"    Max:     {phase_stats['max']:.2f}s")
            print(f"    StdDev:  {phase_stats['std']:.2f}s")