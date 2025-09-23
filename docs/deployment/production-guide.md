# AI Architecture Demo - Production Deployment Guide

## Overview

This guide covers deploying the Enterprise AI Architecture Demo platform in production environments. The platform provides local document processing and agentic analytics capabilities with streaming architecture.

## Architecture Components

### Core Services
- **FastAPI Backend**: RESTful API with authentication and documentation
- **Multi-Agent System**: DocumentAnalyzer, BusinessIntelligence, QualityAssurance agents
- **Streaming Engine**: Kafka-based real-time event processing
- **Vector Database**: ChromaDB for semantic search and embeddings
- **MLOps Platform**: MLflow for model management and experiment tracking

### Infrastructure
- **Database**: PostgreSQL for structured data
- **Cache**: Redis for session management and caching
- **Object Storage**: MinIO for document and model storage
- **Message Queue**: Apache Kafka for event streaming
- **Local LLM**: Ollama for on-premises AI inference

## Production Deployment Options

### Option 1: Docker Swarm Deployment

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy the stack
docker stack deploy -c docker-compose.prod.yml ai-architecture

# Scale services
docker service scale ai-architecture_api=3
docker service scale ai-architecture_agents=2
```

### Option 2: Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ai-architecture

# Access services
kubectl port-forward svc/ai-api 8000:8000 -n ai-architecture
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

See cloud-specific deployment guides in `/deployment/cloud/` directory.

## Pre-Deployment Checklist

### Infrastructure Requirements

**Minimum System Requirements:**
- CPU: 8 cores
- RAM: 32 GB
- Storage: 500 GB SSD
- Network: 1 Gbps

**Recommended for Production:**
- CPU: 16+ cores
- RAM: 64+ GB
- Storage: 1 TB+ NVMe SSD
- Network: 10 Gbps

### Security Configuration

1. **SSL/TLS Certificates**
   ```bash
   # Generate certificates
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
   ```

2. **Environment Variables**
   ```bash
   # Copy and configure production environment
   cp .env.example .env.prod
   
   # Required production variables
   DATABASE_URL="postgresql://user:pass@db:5432/aiarch"
   REDIS_URL="redis://redis:6379/0"
   SECRET_KEY="your-super-secure-secret-key"
   KAFKA_BROKERS="kafka:9092"
   ```

3. **Database Setup**
   ```bash
   # Run migrations
   uv run alembic upgrade head
   
   # Create admin user
   uv run python scripts/create_admin.py
   ```

## Configuration Management

### Environment-Specific Settings

**Development**
```yaml
# config/development.yaml
api:
  debug: true
  log_level: DEBUG
  cors_origins: ["http://localhost:3000"]

database:
  pool_size: 5
  echo: true
```

**Production**
```yaml
# config/production.yaml
api:
  debug: false
  log_level: INFO
  cors_origins: ["https://your-domain.com"]
  
database:
  pool_size: 20
  echo: false
  
security:
  jwt_expire_minutes: 60
  bcrypt_rounds: 12
```

### Secret Management

**Using Docker Secrets**
```bash
# Create secrets
echo "your-db-password" | docker secret create db_password -
echo "your-jwt-secret" | docker secret create jwt_secret -
```

**Using Kubernetes Secrets**
```bash
# Create secret
kubectl create secret generic ai-secrets \
  --from-literal=db-password=your-db-password \
  --from-literal=jwt-secret=your-jwt-secret
```

## Monitoring & Observability

### Health Checks

```bash
# API health check
curl -f http://localhost:8000/health || exit 1

# Database connectivity
curl -f http://localhost:8000/health/db || exit 1

# Kafka connectivity
curl -f http://localhost:8000/health/kafka || exit 1
```

### Metrics Collection

**Prometheus Configuration**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-architecture'
    static_configs:
      - targets: ['api:8000', 'agents:8001']
    metrics_path: /metrics
```

**Grafana Dashboards**
- Import dashboards from `/monitoring/grafana/dashboards/`
- Configure data sources for Prometheus and PostgreSQL

### Logging

**Centralized Logging with ELK Stack**
```bash
# Deploy ELK stack
docker-compose -f docker-compose.logging.yml up -d

# Configure log shipping
# See filebeat.yml configuration
```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

### Kafka Tuning

```properties
# server.properties
num.network.threads=8
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
log.retention.hours=168
log.segment.bytes=1073741824
```

### Redis Optimization

```
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 0
```

## Backup & Recovery

### Database Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h localhost -U postgres -d aiarch > "$BACKUP_DIR/aiarch_$DATE.sql"

# Compress and clean old backups
gzip "$BACKUP_DIR/aiarch_$DATE.sql"
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

### Object Storage Backups

```bash
# MinIO backup using mc client
mc mirror --overwrite minio/documents s3/backup-bucket/documents/
mc mirror --overwrite minio/models s3/backup-bucket/models/
```

### Kafka Topic Backups

```bash
# Export Kafka topics
kafka-console-consumer --bootstrap-server kafka:9092 \
  --topic document-events --from-beginning \
  > kafka-backup-$(date +%Y%m%d).json
```

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Increase container memory limits
docker service update --limit-memory 8g ai-architecture_api
```

**2. Database Connection Issues**
```bash
# Check connection pool status
uv run python -c "from ai_architect_demo.core.database import DatabaseManager; print(DatabaseManager().get_pool_status())"
```

**3. Kafka Consumer Lag**
```bash
# Check consumer group lag
kafka-consumer-groups --bootstrap-server kafka:9092 --describe --group ai-agents
```

### Log Analysis

```bash
# Real-time log monitoring
docker logs -f ai-architecture_api.1.$(docker service ps -q ai-architecture_api)

# Search logs for errors
docker logs ai-architecture_api.1 2>&1 | grep -i error

# Structured log queries
journalctl -u ai-architecture -f --output json | jq '.message'
```

## Security Hardening

### Network Security

```bash
# Configure firewall rules
ufw allow 22/tcp
ufw allow 443/tcp
ufw allow 80/tcp
ufw deny 8000/tcp  # Block direct API access
ufw enable
```

### Container Security

```dockerfile
# Use non-root user
FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Security scanning
RUN apt-get update && apt-get install -y \
    security-updates \
    && rm -rf /var/lib/apt/lists/*
```

### API Security

```python
# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/documents/upload")
@limiter.limit("10/minute")
async def upload_document(request: Request):
    pass
```

## Scaling Guidelines

### Horizontal Scaling

```bash
# Scale API servers
docker service scale ai-architecture_api=5

# Scale agent workers
docker service scale ai-architecture_agents=3

# Add Kafka partitions
kafka-topics --bootstrap-server kafka:9092 \
  --alter --topic document-events --partitions 6
```

### Vertical Scaling

```bash
# Update resource limits
docker service update \
  --limit-memory 16g \
  --limit-cpus 8 \
  ai-architecture_api
```

### Auto-scaling (Kubernetes)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Update system packages
apt update && apt upgrade -y

# Clean Docker system
docker system prune -f

# Rotate logs
logrotate /etc/logrotate.d/ai-architecture

# Vacuum PostgreSQL
psql -d aiarch -c "VACUUM ANALYZE;"

# Restart services if needed
docker service update --force ai-architecture_api
```

### Version Updates

```bash
# Rolling update
docker service update --image ai-architecture:v1.1.0 ai-architecture_api

# Rollback if needed
docker service rollback ai-architecture_api
```

## Support & Documentation

- **API Documentation**: `http://your-domain/docs`
- **Monitoring Dashboard**: `http://your-domain:3000` (Grafana)
- **Log Analysis**: `http://your-domain:5601` (Kibana)
- **Issue Tracking**: See GitHub repository
- **Community Support**: Join our Discord server

For additional support, please contact the development team or consult the troubleshooting guide.