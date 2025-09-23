#!/bin/bash

# AI Architecture Deployment Script for Docker Swarm
# This script deploys the complete AI architecture stack using Docker Swarm

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="ai-architecture"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check requirements
check_requirements() {
    log "Checking deployment requirements..."
    
    # Check Docker Swarm
    if ! docker info | grep -q "Swarm: active"; then
        error "Docker Swarm is not initialized. Run: docker swarm init"
    fi
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Compose file $COMPOSE_FILE not found"
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file $ENV_FILE not found. Creating template..."
        create_env_template
    fi
    
    log "Requirements check passed"
}

# Create environment template
create_env_template() {
    cat > "$ENV_FILE" << EOF
# Production Environment Configuration
# IMPORTANT: Change all passwords and keys before deployment!

# Database Configuration
POSTGRES_DB=ai_architecture
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=change_this_password_123!
DATABASE_URL=postgresql://ai_user:change_this_password_123!@postgres:5432/ai_architecture

# Redis Configuration
REDIS_URL=redis://redis:6379

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=change_this_password_456!

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Security Configuration
SECRET_KEY=change_this_secret_key_very_long_and_random_string_789!
JWT_SECRET_KEY=change_this_jwt_secret_key_also_very_long_and_random!

# Monitoring Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_this_password_grafana!

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=3

# Agent Configuration
AGENT_WORKERS=2

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# SSL Configuration (if using HTTPS)
SSL_CERT_PATH=/etc/ssl/certs/ai-architecture.crt
SSL_KEY_PATH=/etc/ssl/private/ai-architecture.key

# External URLs (update with your domain)
EXTERNAL_API_URL=https://api.your-domain.com
EXTERNAL_GRAFANA_URL=https://grafana.your-domain.com
EXTERNAL_MLFLOW_URL=https://mlflow.your-domain.com
EOF
    
    warn "Environment template created at $ENV_FILE"
    warn "Please update all passwords and configuration before deployment!"
    read -p "Press Enter to continue after updating the environment file..."
}

# Create Docker secrets
create_secrets() {
    log "Creating Docker secrets..."
    
    # Read environment variables
    source "$ENV_FILE"
    
    # Create secrets if they don't exist
    echo "$POSTGRES_PASSWORD" | docker secret create postgres_password - 2>/dev/null || true
    echo "$MINIO_ROOT_PASSWORD" | docker secret create minio_password - 2>/dev/null || true
    echo "$SECRET_KEY" | docker secret create app_secret_key - 2>/dev/null || true
    echo "$JWT_SECRET_KEY" | docker secret create jwt_secret_key - 2>/dev/null || true
    echo "$GRAFANA_ADMIN_PASSWORD" | docker secret create grafana_password - 2>/dev/null || true
    
    log "Docker secrets created"
}

# Create Docker networks
create_networks() {
    log "Creating Docker networks..."
    
    docker network create --driver overlay --attachable ai-architecture-network 2>/dev/null || true
    docker network create --driver overlay --attachable ai-monitoring 2>/dev/null || true
    
    log "Docker networks created"
}

# Deploy the stack
deploy_stack() {
    log "Deploying AI Architecture stack..."
    
    # Deploy with environment file
    docker stack deploy \
        --compose-file "$COMPOSE_FILE" \
        --with-registry-auth \
        "$STACK_NAME"
    
    log "Stack deployment initiated"
}

# Wait for services
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        local ready_services=$(docker stack services "$STACK_NAME" --format "table {{.Name}}\t{{.Replicas}}" | grep -c "1/1" || echo "0")
        local total_services=$(docker stack services "$STACK_NAME" --quiet | wc -l)
        
        log "Services ready: $ready_services/$total_services (attempt $attempt/$max_attempts)"
        
        if [ "$ready_services" -eq "$total_services" ] && [ "$total_services" -gt 0 ]; then
            log "All services are ready!"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    warn "Some services may still be starting. Check with: docker stack services $STACK_NAME"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    
    echo "Stack Services:"
    docker stack services "$STACK_NAME"
    echo ""
    
    echo "Service Endpoints:"
    echo "• API: http://localhost:8000"
    echo "• API Docs: http://localhost:8000/docs"
    echo "• Grafana: http://localhost:3000 (admin/\$GRAFANA_ADMIN_PASSWORD)"
    echo "• MLflow: http://localhost:5000"
    echo "• MinIO Console: http://localhost:9001 (minioadmin/\$MINIO_ROOT_PASSWORD)"
    echo "• Prometheus: http://localhost:9090"
    echo ""
    
    echo "Useful Commands:"
    echo "• View logs: docker service logs \${STACK_NAME}_\${SERVICE_NAME}"
    echo "• Scale service: docker service scale \${STACK_NAME}_\${SERVICE_NAME}=N"
    echo "• Update stack: docker stack deploy -c $COMPOSE_FILE $STACK_NAME"
    echo "• Remove stack: docker stack rm $STACK_NAME"
    echo ""
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local endpoints=(
        "http://localhost:8000/health"
        "http://localhost:3000/api/health"
        "http://localhost:9090/-/healthy"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" >/dev/null 2>&1; then
            log "✓ $endpoint is healthy"
        else
            warn "✗ $endpoint is not responding"
        fi
    done
}

# Backup function
backup_data() {
    log "Creating backup..."
    
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    docker run --rm -v ai-architecture_postgres_data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/postgres_data.tar.gz -C /data .
    docker run --rm -v ai-architecture_mlflow_data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/mlflow_data.tar.gz -C /data .
    docker run --rm -v ai-architecture_minio_data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/minio_data.tar.gz -C /data .
    
    # Backup configuration
    cp "$ENV_FILE" "$backup_dir/"
    cp "$COMPOSE_FILE" "$backup_dir/"
    
    log "Backup created at $backup_dir"
}

# Main deployment function
main() {
    log "Starting AI Architecture deployment on Docker Swarm"
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_requirements
            create_secrets
            create_networks
            deploy_stack
            wait_for_services
            show_status
            health_check
            ;;
        "status")
            show_status
            ;;
        "backup")
            backup_data
            ;;
        "remove")
            log "Removing AI Architecture stack..."
            docker stack rm "$STACK_NAME"
            log "Stack removal initiated"
            ;;
        "logs")
            if [ -n "$2" ]; then
                docker service logs "${STACK_NAME}_$2"
            else
                error "Please specify service name: $0 logs <service_name>"
            fi
            ;;
        "scale")
            if [ -n "$2" ] && [ -n "$3" ]; then
                log "Scaling ${STACK_NAME}_$2 to $3 replicas"
                docker service scale "${STACK_NAME}_$2=$3"
            else
                error "Usage: $0 scale <service_name> <replica_count>"
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|status|backup|remove|logs|scale}"
            echo ""
            echo "Commands:"
            echo "  deploy  - Deploy the complete stack"
            echo "  status  - Show deployment status"
            echo "  backup  - Create backup of data"
            echo "  remove  - Remove the stack"
            echo "  logs    - Show logs for a service"
            echo "  scale   - Scale a service"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"