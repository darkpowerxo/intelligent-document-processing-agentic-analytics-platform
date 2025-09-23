# API Documentation

This directory contains the complete API reference documentation for the AI Architecture Demo platform.

## Documentation Structure

| File | Description |
|------|-------------|
| [authentication.md](authentication.md) | Authentication and authorization |
| [documents.md](documents.md) | Document processing endpoints |
| [agents.md](agents.md) | AI agent management endpoints |
| [analytics.md](analytics.md) | Business intelligence and analytics |
| [events.md](events.md) | Event streaming and WebSocket APIs |
| [system.md](system.md) | System management and monitoring |
| [webhooks.md](webhooks.md) | Webhook configuration and callbacks |
| [errors.md](errors.md) | Error codes and troubleshooting |

## Quick Links

- **Base URL**: `http://localhost:8000` (development)
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## Getting Started

1. [Authentication Guide](authentication.md) - Get your API token
2. [Quick Start Examples](examples/) - Common use cases
3. [SDK Documentation](sdks/) - Client libraries
4. [Postman Collection](postman/) - API testing collection

## API Versioning

The API uses semantic versioning with the version specified in the URL:
- Current version: `v1`
- Base path: `/api/v1/`
- Deprecation policy: 12 months notice for breaking changes

## Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|--------|
| Authentication | 10 requests | 1 minute |
| Document Upload | 50 requests | 1 hour |
| General API | 1000 requests | 1 hour |
| Streaming | 1 connection | per API key |

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/ai-architect-demo/issues)
- **Documentation**: This directory
- **Examples**: [examples/](examples/) directory