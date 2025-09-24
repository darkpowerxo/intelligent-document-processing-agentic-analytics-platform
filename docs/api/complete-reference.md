# AI Architecture API Documentation

## Overview

This document provides comprehensive documentation for the AI Architecture API, including authentication, endpoints, data models, and usage examples.

## Table of Contents

1. [Base Information](#base-information)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Event Streaming](#event-streaming)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [SDKs and Examples](#sdks-and-examples)

## Base Information

- **Base URL**: `http://localhost:8002` (development) / `https://api.your-domain.com` (production)
- **API Version**: v1
- **Content Type**: `application/json`
- **Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## Authentication

### JWT Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Obtaining a Token

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Token Refresh

```http
POST /api/v1/auth/refresh
Authorization: Bearer <your-refresh-token>
```

## API Endpoints

### Health Check

#### Get System Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "kafka": "healthy",
    "mlflow": "healthy"
  }
}
```

### Document Processing

#### Upload Document
```http
POST /api/v1/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <document-file>
metadata: {
  "category": "contract",
  "priority": "high",
  "tags": ["legal", "urgent"]
}
```

Response:
```json
{
  "document_id": "doc_123456789",
  "filename": "contract.pdf",
  "size": 1048576,
  "upload_time": "2024-01-15T10:30:00Z",
  "status": "uploaded",
  "processing_queue": "document_analysis"
}
```

#### Get Document Status
```http
GET /api/v1/documents/{document_id}/status
Authorization: Bearer <token>
```

Response:
```json
{
  "document_id": "doc_123456789",
  "status": "processed",
  "processing_stages": {
    "upload": "completed",
    "analysis": "completed",
    "extraction": "completed",
    "quality_check": "completed"
  },
  "results_available": true,
  "processing_time": 45.2
}
```

#### Get Document Analysis
```http
GET /api/v1/documents/{document_id}/analysis
Authorization: Bearer <token>
```

Response:
```json
{
  "document_id": "doc_123456789",
  "analysis": {
    "document_type": "contract",
    "confidence": 0.95,
    "key_entities": [
      {
        "type": "person",
        "value": "John Smith",
        "confidence": 0.98,
        "position": {"start": 150, "end": 160}
      }
    ],
    "sentiment": {
      "overall": "neutral",
      "score": 0.1,
      "sections": [
        {"section": "terms", "sentiment": "positive", "score": 0.3}
      ]
    },
    "summary": "This contract establishes terms for service delivery...",
    "quality_score": 0.92,
    "recommendations": [
      "Review clause 3.2 for completeness",
      "Consider adding termination conditions"
    ]
  },
  "metadata": {
    "processed_at": "2024-01-15T10:35:00Z",
    "processing_duration": 45.2,
    "agent_version": "v2.1.0"
  }
}
```

### Agent Management

#### List Available Agents
```http
GET /api/v1/agents/
Authorization: Bearer <token>
```

Response:
```json
{
  "agents": [
    {
      "id": "document_analyzer",
      "name": "Document Analyzer Agent",
      "status": "active",
      "capabilities": ["text_extraction", "entity_recognition", "classification"],
      "current_load": 0.65,
      "queue_length": 12
    },
    {
      "id": "business_intelligence",
      "name": "Business Intelligence Agent",
      "status": "active",
      "capabilities": ["data_analysis", "reporting", "insights"],
      "current_load": 0.43,
      "queue_length": 8
    }
  ]
}
```

#### Get Agent Status
```http
GET /api/v1/agents/{agent_id}/status
Authorization: Bearer <token>
```

Response:
```json
{
  "agent_id": "document_analyzer",
  "name": "Document Analyzer Agent",
  "status": "active",
  "health": "healthy",
  "metrics": {
    "tasks_completed": 1547,
    "success_rate": 0.98,
    "avg_processing_time": 12.5,
    "current_queue_length": 12,
    "load_percentage": 65
  },
  "last_heartbeat": "2024-01-15T10:34:55Z"
}
```

#### Submit Task to Agent
```http
POST /api/v1/agents/{agent_id}/tasks
Authorization: Bearer <token>
Content-Type: application/json

{
  "task_type": "document_analysis",
  "data": {
    "document_id": "doc_123456789",
    "analysis_options": {
      "include_entities": true,
      "include_sentiment": true,
      "include_summary": true
    }
  },
  "priority": "high",
  "callback_url": "https://your-app.com/callbacks/task-complete"
}
```

Response:
```json
{
  "task_id": "task_987654321",
  "agent_id": "document_analyzer",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:40:00Z",
  "queue_position": 3
}
```

### Business Intelligence

#### Get Analytics Dashboard Data
```http
GET /api/v1/analytics/dashboard
Authorization: Bearer <token>
Query Parameters:
  - start_date: 2024-01-01
  - end_date: 2024-01-15
  - metric_types: documents,agents,performance
```

Response:
```json
{
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "metrics": {
    "documents_processed": 1247,
    "success_rate": 0.97,
    "avg_processing_time": 15.3,
    "agent_utilization": 0.72
  },
  "trends": {
    "daily_volume": [85, 92, 78, 95, 103, 87, 91],
    "processing_times": [14.2, 15.1, 16.0, 14.8, 15.5, 15.2, 15.3],
    "error_rates": [0.03, 0.02, 0.04, 0.02, 0.01, 0.03, 0.02]
  },
  "insights": [
    {
      "type": "performance",
      "message": "Processing time improved by 8% this week",
      "severity": "info"
    }
  ]
}
```

#### Generate Custom Report
```http
POST /api/v1/analytics/reports
Authorization: Bearer <token>
Content-Type: application/json

{
  "report_type": "performance_summary",
  "parameters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-15"
    },
    "filters": {
      "document_types": ["contract", "invoice"],
      "agents": ["document_analyzer", "business_intelligence"]
    },
    "metrics": ["processing_time", "success_rate", "queue_length"],
    "format": "detailed"
  },
  "delivery": {
    "method": "email",
    "recipients": ["admin@company.com"],
    "schedule": "immediate"
  }
}
```

### Event Streaming

#### Subscribe to Event Stream
```http
GET /api/v1/events/stream
Authorization: Bearer <token>
Query Parameters:
  - event_types: document_processed,agent_status,system_alert
  - format: sse
```

Server-Sent Events Response:
```
data: {"event_type": "document_processed", "document_id": "doc_123", "status": "completed", "timestamp": "2024-01-15T10:30:00Z"}

data: {"event_type": "agent_status", "agent_id": "document_analyzer", "status": "active", "load": 0.65, "timestamp": "2024-01-15T10:30:05Z"}
```

#### Get Event History
```http
GET /api/v1/events/history
Authorization: Bearer <token>
Query Parameters:
  - start_time: 2024-01-15T09:00:00Z
  - end_time: 2024-01-15T11:00:00Z
  - event_types: document_processed,system_alert
  - limit: 100
  - offset: 0
```

Response:
```json
{
  "events": [
    {
      "id": "evt_123456789",
      "event_type": "document_processed",
      "timestamp": "2024-01-15T10:30:00Z",
      "data": {
        "document_id": "doc_123456789",
        "status": "completed",
        "processing_time": 45.2
      }
    }
  ],
  "pagination": {
    "total": 1247,
    "limit": 100,
    "offset": 0,
    "has_next": true
  }
}
```

### System Management

#### Get System Metrics
```http
GET /api/v1/system/metrics
Authorization: Bearer <token>
```

Response:
```json
{
  "system": {
    "cpu_usage": 0.65,
    "memory_usage": 0.72,
    "disk_usage": 0.45,
    "uptime": 7200
  },
  "database": {
    "connections": 15,
    "query_time_avg": 23.5,
    "slow_queries": 2
  },
  "kafka": {
    "topics": 6,
    "partitions": 18,
    "consumers": 8,
    "lag_total": 245
  },
  "redis": {
    "memory_used": "256MB",
    "keys": 1547,
    "hit_rate": 0.94
  }
}
```

#### Update System Configuration
```http
PUT /api/v1/system/config
Authorization: Bearer <token>
Content-Type: application/json

{
  "settings": {
    "max_concurrent_tasks": 50,
    "task_timeout": 300,
    "log_level": "INFO",
    "feature_flags": {
      "enhanced_analysis": true,
      "real_time_streaming": true
    }
  }
}
```

## Data Models

### Document
```json
{
  "id": "string",
  "filename": "string",
  "content_type": "string",
  "size": "integer",
  "upload_time": "datetime",
  "status": "uploaded|processing|completed|failed",
  "metadata": {
    "category": "string",
    "tags": ["string"],
    "priority": "low|medium|high"
  },
  "analysis_results": {
    "document_type": "string",
    "confidence": "float",
    "entities": ["Entity"],
    "sentiment": "Sentiment",
    "summary": "string",
    "quality_score": "float"
  }
}
```

### Agent
```json
{
  "id": "string",
  "name": "string",
  "type": "document_analyzer|business_intelligence|quality_assurance",
  "status": "active|inactive|error",
  "capabilities": ["string"],
  "configuration": {
    "max_concurrent_tasks": "integer",
    "timeout": "integer",
    "model_version": "string"
  },
  "metrics": {
    "tasks_completed": "integer",
    "success_rate": "float",
    "avg_processing_time": "float",
    "queue_length": "integer"
  }
}
```

### Task
```json
{
  "id": "string",
  "agent_id": "string",
  "type": "string",
  "status": "queued|processing|completed|failed|cancelled",
  "priority": "low|medium|high",
  "created_at": "datetime",
  "started_at": "datetime",
  "completed_at": "datetime",
  "data": "object",
  "results": "object",
  "error_message": "string",
  "retry_count": "integer"
}
```

### Event
```json
{
  "id": "string",
  "event_type": "string",
  "timestamp": "datetime",
  "source": "string",
  "data": "object",
  "correlation_id": "string",
  "metadata": {
    "version": "string",
    "environment": "string"
  }
}
```

## Event Streaming

The API provides real-time event streaming capabilities using Server-Sent Events (SSE) and Kafka integration.

### Event Types

- `document_uploaded`: New document uploaded
- `document_processed`: Document processing completed
- `agent_status_changed`: Agent status update
- `task_completed`: Task finished processing
- `system_alert`: System-wide alerts and warnings
- `performance_threshold`: Performance metrics exceeded thresholds

### Event Format

All events follow a consistent structure:
```json
{
  "event_id": "unique-event-id",
  "event_type": "event_name",
  "timestamp": "2024-01-15T10:30:00Z",
  "source": "service_name",
  "data": {
    // Event-specific data
  },
  "correlation_id": "request-correlation-id",
  "metadata": {
    "version": "1.0",
    "environment": "production"
  }
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error information:

### Error Response Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Specific error details"
    },
    "correlation_id": "request-correlation-id",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Examples

#### Validation Error
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "filename": "Filename is required",
      "file_size": "File size exceeds maximum limit of 10MB"
    },
    "correlation_id": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Rate Limit Error
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "window": "60s",
      "retry_after": 45
    },
    "correlation_id": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default**: 100 requests per minute per API key
- **Upload**: 10 file uploads per minute per API key
- **Streaming**: 1 concurrent stream per API key
- **Reports**: 5 report generations per hour per API key

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
X-RateLimit-Window: 60
```

## SDKs and Examples

### Python SDK Example

```python
import asyncio
from ai_architecture_sdk import AIArchitectureClient, Document

# Initialize client
client = AIArchitectureClient(
    base_url="http://localhost:8002",
    api_key="your-api-key"
)

async def process_document():
    # Upload document
    with open("contract.pdf", "rb") as f:
        document = await client.documents.upload(
            file=f,
            metadata={
                "category": "contract",
                "priority": "high",
                "tags": ["legal", "urgent"]
            }
        )
    
    print(f"Document uploaded: {document.id}")
    
    # Wait for processing
    result = await client.documents.wait_for_completion(
        document.id,
        timeout=300
    )
    
    if result.status == "completed":
        analysis = await client.documents.get_analysis(document.id)
        print(f"Document type: {analysis.document_type}")
        print(f"Summary: {analysis.summary}")
    else:
        print(f"Processing failed: {result.error_message}")

# Run async function
asyncio.run(process_document())
```

### JavaScript SDK Example

```javascript
import { AIArchitectureClient } from '@ai-architecture/sdk';

const client = new AIArchitectureClient({
  baseURL: 'http://localhost:8002',
  apiKey: 'your-api-key'
});

async function processDocument() {
  try {
    // Upload document
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('metadata', JSON.stringify({
      category: 'contract',
      priority: 'high',
      tags: ['legal', 'urgent']
    }));
    
    const document = await client.documents.upload(formData);
    console.log(`Document uploaded: ${document.id}`);
    
    // Subscribe to status updates
    const eventSource = client.events.subscribe({
      eventTypes: ['document_processed'],
      filter: { document_id: document.id }
    });
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.document_id === document.id && data.status === 'completed') {
        console.log('Document processing completed');
        // Get analysis results
        client.documents.getAnalysis(document.id)
          .then(analysis => {
            console.log(`Document type: ${analysis.document_type}`);
            console.log(`Summary: ${analysis.summary}`);
          });
      }
    };
    
  } catch (error) {
    console.error('Error processing document:', error);
  }
}
```

### cURL Examples

#### Upload Document
```bash
curl -X POST "http://localhost:8002/api/v1/documents/upload" \
  -H "Authorization: Bearer your-jwt-token" \
  -F "file=@contract.pdf" \
  -F "metadata={\"category\":\"contract\",\"priority\":\"high\"}"
```

#### Get Document Analysis
```bash
curl -X GET "http://localhost:8002/api/v1/documents/doc_123456789/analysis" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Accept: application/json"
```

#### Stream Events
```bash
curl -X GET "http://localhost:8002/api/v1/events/stream?event_types=document_processed" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Accept: text/event-stream"
```

## WebSocket Support

For real-time bidirectional communication, the API also supports WebSocket connections:

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/events');

ws.onopen = function() {
  // Subscribe to events
  ws.send(JSON.stringify({
    action: 'subscribe',
    event_types: ['document_processed', 'agent_status'],
    auth_token: 'your-jwt-token'
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received event:', data);
};
```

## Monitoring and Observability

The API provides comprehensive monitoring endpoints:

### Metrics Endpoint
```http
GET /metrics
```
Returns Prometheus-compatible metrics for monitoring and alerting.

### Health Check Endpoints
- `/health` - Basic health check
- `/health/detailed` - Detailed health information
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

For more information and advanced usage examples, visit the interactive API documentation at `/docs` when running the service.