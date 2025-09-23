#!/bin/bash

# AI Architecture Deployment Script for Kubernetes
# This script deploys the complete AI architecture stack on Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ai-architecture"
K8S_DIR="k8s"
ENV_FILE=".env.k8s"

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
    log "Checking Kubernetes deployment requirements..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install kubectl."
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
    fi
    
    # Check k8s directory
    if [ ! -d "$K8S_DIR" ]; then
        error "Kubernetes manifests directory $K8S_DIR not found"
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file $ENV_FILE not found. Creating template..."
        create_env_template
    fi
    
    log "Requirements check passed"
}

# Create environment template for Kubernetes
create_env_template() {
    cat > "$ENV_FILE" << EOF
# Kubernetes Environment Configuration
# IMPORTANT: Update these values before deployment!

# Base64 encoded secrets (use: echo -n 'password' | base64)
POSTGRES_DB=$(echo -n 'ai_architecture' | base64)
POSTGRES_USER=$(echo -n 'ai_user' | base64)
POSTGRES_PASSWORD=$(echo -n 'change_this_password_123!' | base64)

MINIO_ACCESS_KEY=$(echo -n 'minioadmin' | base64)
MINIO_SECRET_KEY=$(echo -n 'change_this_password_456!' | base64)

SECRET_KEY=$(echo -n 'change_this_secret_key_very_long_and_random_string_789!' | base64)
JWT_SECRET_KEY=$(echo -n 'change_this_jwt_secret_key_also_very_long_and_random!' | base64)

GRAFANA_ADMIN_USER=$(echo -n 'admin' | base64)
GRAFANA_ADMIN_PASSWORD=$(echo -n 'change_this_password_grafana!' | base64)

# Storage class (update based on your cluster)
STORAGE_CLASS=gp2

# Domain configuration (update with your domain)
DOMAIN=your-domain.com
API_SUBDOMAIN=api
GRAFANA_SUBDOMAIN=grafana
MLFLOW_SUBDOMAIN=mlflow

# TLS configuration
ENABLE_TLS=true
TLS_SECRET_NAME=ai-architecture-tls

# Resource configuration
API_REPLICAS=3
AGENT_REPLICAS=2
STREAMING_REPLICAS=2

# Node affinity (optional)
NODE_SELECTOR=""
EOF
    
    warn "Environment template created at $ENV_FILE"
    warn "Please update all passwords and configuration before deployment!"
    read -p "Press Enter to continue after updating the environment file..."
}

# Create namespace
create_namespace() {
    log "Creating namespace..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    log "Namespace created/updated"
}

# Create secrets
create_secrets() {
    log "Creating Kubernetes secrets..."
    
    # Source environment variables
    source "$ENV_FILE"
    
    # Create secrets manifest
    cat > /tmp/secrets.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: ai-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  postgres-db: $POSTGRES_DB
  postgres-user: $POSTGRES_USER
  postgres-password: $POSTGRES_PASSWORD
  minio-access-key: $MINIO_ACCESS_KEY
  minio-secret-key: $MINIO_SECRET_KEY
  secret-key: $SECRET_KEY
  jwt-secret-key: $JWT_SECRET_KEY
  grafana-admin-user: $GRAFANA_ADMIN_USER
  grafana-admin-password: $GRAFANA_ADMIN_PASSWORD
EOF
    
    kubectl apply -f /tmp/secrets.yaml
    rm /tmp/secrets.yaml
    
    log "Secrets created"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log "Deploying infrastructure services..."
    
    # Apply infrastructure manifests
    kubectl apply -f "$K8S_DIR/infrastructure.yaml"
    
    log "Infrastructure deployment initiated"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    kubectl apply -f "$K8S_DIR/monitoring.yaml"
    
    log "Monitoring deployment initiated"
}

# Deploy application services
deploy_applications() {
    log "Deploying application services..."
    
    kubectl apply -f "$K8S_DIR/ai-architecture.yaml"
    
    log "Application deployment initiated"
}

# Wait for deployments
wait_for_deployments() {
    log "Waiting for deployments to be ready..."
    
    local deployments=(
        "postgres"
        "redis"
        "kafka"
        "minio"
        "mlflow"
        "ai-api"
        "ai-agent-worker"
        "streaming-processor"
        "prometheus"
        "grafana"
    )
    
    for deployment in "${deployments[@]}"; do
        log "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n "$NAMESPACE" 2>/dev/null || \
        kubectl wait --for=condition=ready --timeout=300s statefulset/$deployment -n "$NAMESPACE" 2>/dev/null || \
        warn "Timeout waiting for $deployment"
    done
    
    log "Deployments ready"
}

# Show deployment status
show_status() {
    log "Kubernetes Deployment Status:"
    echo ""
    
    echo "Namespace: $NAMESPACE"
    kubectl get namespace "$NAMESPACE"
    echo ""
    
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    echo ""
    
    echo "Services:"
    kubectl get services -n "$NAMESPACE"
    echo ""
    
    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress configured"
    echo ""
    
    echo "Persistent Volume Claims:"
    kubectl get pvc -n "$NAMESPACE"
    echo ""
    
    # Show service endpoints
    local api_port=$(kubectl get service ai-api -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "N/A")
    local grafana_port=$(kubectl get service grafana -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "N/A")
    local mlflow_port=$(kubectl get service mlflow -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "N/A")
    
    echo "Service Access (if using NodePort):"
    echo "• API: http://\${NODE_IP}:$api_port"
    echo "• API Docs: http://\${NODE_IP}:$api_port/docs"
    echo "• Grafana: http://\${NODE_IP}:$grafana_port"
    echo "• MLflow: http://\${NODE_IP}:$mlflow_port"
    echo ""
    
    echo "Port Forward Commands:"
    echo "• kubectl port-forward -n $NAMESPACE service/ai-api 8000:8000"
    echo "• kubectl port-forward -n $NAMESPACE service/grafana 3000:3000"
    echo "• kubectl port-forward -n $NAMESPACE service/mlflow 5000:5000"
    echo "• kubectl port-forward -n $NAMESPACE service/prometheus 9090:9090"
    echo ""
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Check pod status
    local failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running -o name 2>/dev/null | wc -l)
    if [ "$failed_pods" -eq 0 ]; then
        log "✓ All pods are running"
    else
        warn "✗ $failed_pods pods are not running"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
    fi
    
    # Check service endpoints
    log "Checking service endpoints..."
    kubectl get endpoints -n "$NAMESPACE" | grep -v "^NAME" | while read line; do
        local service=$(echo "$line" | awk '{print $1}')
        local endpoints=$(echo "$line" | awk '{print $2}')
        
        if [ "$endpoints" != "<none>" ]; then
            log "✓ $service has endpoints"
        else
            warn "✗ $service has no endpoints"
        fi
    done
}

# Setup port forwarding
setup_port_forward() {
    log "Setting up port forwarding..."
    
    # Kill existing port forwards
    pkill -f "kubectl port-forward" 2>/dev/null || true
    sleep 2
    
    # Setup port forwards in background
    kubectl port-forward -n "$NAMESPACE" service/ai-api 8000:8000 &
    kubectl port-forward -n "$NAMESPACE" service/grafana 3000:3000 &
    kubectl port-forward -n "$NAMESPACE" service/mlflow 5000:5000 &
    kubectl port-forward -n "$NAMESPACE" service/prometheus 9090:9090 &
    
    log "Port forwarding setup complete"
    log "Services available at:"
    log "• API: http://localhost:8000"
    log "• Grafana: http://localhost:3000"
    log "• MLflow: http://localhost:5000"
    log "• Prometheus: http://localhost:9090"
}

# Backup function
backup_data() {
    log "Creating backup..."
    
    local backup_dir="./backups/k8s_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup manifests
    cp -r "$K8S_DIR" "$backup_dir/"
    cp "$ENV_FILE" "$backup_dir/"
    
    # Backup cluster state
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/cluster-state.yaml"
    kubectl get pvc -n "$NAMESPACE" -o yaml > "$backup_dir/persistent-volumes.yaml"
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml"
    
    log "Backup created at $backup_dir"
}

# Cleanup function
cleanup() {
    log "Cleaning up AI Architecture deployment..."
    
    # Delete applications
    kubectl delete -f "$K8S_DIR/ai-architecture.yaml" 2>/dev/null || true
    
    # Delete monitoring
    kubectl delete -f "$K8S_DIR/monitoring.yaml" 2>/dev/null || true
    
    # Delete infrastructure
    kubectl delete -f "$K8S_DIR/infrastructure.yaml" 2>/dev/null || true
    
    # Wait for pods to terminate
    log "Waiting for pods to terminate..."
    kubectl wait --for=delete pods --all -n "$NAMESPACE" --timeout=300s 2>/dev/null || true
    
    # Delete PVCs
    kubectl delete pvc --all -n "$NAMESPACE" 2>/dev/null || true
    
    # Delete namespace
    kubectl delete namespace "$NAMESPACE" 2>/dev/null || true
    
    log "Cleanup completed"
}

# Main function
main() {
    log "Starting AI Architecture deployment on Kubernetes"
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_requirements
            create_namespace
            create_secrets
            deploy_infrastructure
            deploy_monitoring
            deploy_applications
            wait_for_deployments
            show_status
            health_check
            ;;
        "status")
            show_status
            health_check
            ;;
        "port-forward")
            setup_port_forward
            ;;
        "backup")
            backup_data
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            if [ -n "$2" ]; then
                kubectl logs -n "$NAMESPACE" -l app="$2" --tail=100 -f
            else
                error "Please specify app label: $0 logs <app_name>"
            fi
            ;;
        "shell")
            if [ -n "$2" ]; then
                local pod=$(kubectl get pods -n "$NAMESPACE" -l app="$2" -o jsonpath='{.items[0].metadata.name}')
                if [ -n "$pod" ]; then
                    kubectl exec -it -n "$NAMESPACE" "$pod" -- /bin/bash
                else
                    error "No pod found for app: $2"
                fi
            else
                error "Please specify app label: $0 shell <app_name>"
            fi
            ;;
        "scale")
            if [ -n "$2" ] && [ -n "$3" ]; then
                log "Scaling $2 to $3 replicas"
                kubectl scale deployment "$2" --replicas="$3" -n "$NAMESPACE"
            else
                error "Usage: $0 scale <deployment_name> <replica_count>"
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|status|port-forward|backup|cleanup|logs|shell|scale}"
            echo ""
            echo "Commands:"
            echo "  deploy       - Deploy the complete stack"
            echo "  status       - Show deployment status"
            echo "  port-forward - Setup port forwarding for services"
            echo "  backup       - Create backup of configuration"
            echo "  cleanup      - Remove the complete deployment"
            echo "  logs         - Show logs for an app"
            echo "  shell        - Get shell access to a pod"
            echo "  scale        - Scale a deployment"
            exit 1
            ;;
    esac
}

# Handle script termination
trap 'log "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"