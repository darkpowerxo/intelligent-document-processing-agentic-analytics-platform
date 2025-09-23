# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) that document the key architectural decisions made during the development of the AI Architecture Demo platform.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-multi-agent-architecture.md) | Multi-Agent Architecture Pattern | Accepted | 2024-01-15 |
| [ADR-002](002-event-driven-architecture.md) | Event-Driven Architecture with Kafka | Accepted | 2024-01-16 |
| [ADR-003](003-containerization-strategy.md) | Docker Containerization Strategy | Accepted | 2024-01-17 |
| [ADR-004](004-database-selection.md) | PostgreSQL as Primary Database | Accepted | 2024-01-18 |
| [ADR-005](005-api-framework-selection.md) | FastAPI for REST API Development | Accepted | 2024-01-19 |
| [ADR-006](006-monitoring-stack.md) | Prometheus and Grafana Monitoring | Accepted | 2024-01-20 |

## ADR Template

When creating new ADRs, use the template provided in [template.md](template.md).

## ADR Status

- **Proposed**: The ADR is under consideration
- **Accepted**: The ADR has been approved and should be implemented
- **Deprecated**: The ADR is no longer relevant but kept for historical context
- **Superseded**: The ADR has been replaced by a newer decision